// chipBLAS — load CLBlast tuner JSONs at handle-bind time so users
// don't need to rebuild CLBlast with the tuned database baked in.
//
// Set CHIPBLAS_TUNING_DIR=<dir> to point at a directory full of files
// named like `clblast_<family>_<id>_<precision>.json` (the format
// emitted by `clblast_tuner_xgemm`, `clblast_tuner_xgemv`, etc.). For
// each file we extract `best_kernel` and `best_parameters` and call
// clblast::OverrideParameters for the bound device.
//
// SPDX-License-Identifier: MIT

#include "chipblas_internal.hh"

#include <clblast.h>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>

namespace chipblas {

namespace {

// Tiny no-deps JSON value extractor. The tuner files are flat enough
// that we don't need a real parser; we look for `"key": "value"` or
// `"key": "12"` patterns at the top level.
std::string extractString(const std::string& blob, const std::string& key) {
    auto needle = "\"" + key + "\"";
    auto p = blob.find(needle);
    if (p == std::string::npos) return {};
    p = blob.find(':', p);
    if (p == std::string::npos) return {};
    auto q = blob.find('"', p);
    if (q == std::string::npos) return {};
    auto r = blob.find('"', q + 1);
    if (r == std::string::npos) return {};
    return blob.substr(q + 1, r - q - 1);
}

// "GEMMK=1 KREG=8 KWG=1 KWI=1 ..." → unordered_map<string,size_t>.
std::unordered_map<std::string, size_t>
parseParameters(const std::string& s) {
    std::unordered_map<std::string, size_t> out;
    std::istringstream iss(s);
    std::string tok;
    while (iss >> tok) {
        auto eq = tok.find('=');
        if (eq == std::string::npos) continue;
        auto key = tok.substr(0, eq);
        auto val = tok.substr(eq + 1);
        try { out[key] = static_cast<size_t>(std::stoul(val)); }
        catch (...) { /* skip malformed entries silently */ }
    }
    return out;
}

clblast::Precision parsePrecision(const std::string& s) {
    if (s == "16")    return clblast::Precision::kHalf;
    if (s == "32")    return clblast::Precision::kSingle;
    if (s == "64")    return clblast::Precision::kDouble;
    if (s == "3232")  return clblast::Precision::kComplexSingle;
    if (s == "6464")  return clblast::Precision::kComplexDouble;
    return clblast::Precision::kSingle;
}

// CLBlast's OverrideParameters keys on the *family* name (the title of
// the database entry: "Copy", "Pad", "Xgemm", "XgemmDirect", etc.). The
// tuner JSONs report `kernel_family` like "xgemm_12" or "xgemm_direct_1"
// and `best_kernel` like "Xgemm" or "XgemmDirectTN" (the kernel
// instantiation that the tuner picked) — neither directly matches what
// the override API wants. Map the family to the database key here.
std::string familyToKey(const std::string& family) {
    // Strip a trailing "_<digits>" suffix that the tuner appends.
    std::string base = family;
    auto u = base.rfind('_');
    if (u != std::string::npos) {
        bool allDigits = true;
        for (size_t i = u + 1; i < base.size(); ++i)
            if (!std::isdigit(static_cast<unsigned char>(base[i]))) {
                allDigits = false; break;
            }
        if (allDigits && u + 1 < base.size()) base = base.substr(0, u);
    }
    if (base == "copy")           return "Copy";
    if (base == "pad")            return "Pad";
    if (base == "transpose")      return "Transpose";
    if (base == "padtranspose")   return "Padtranspose";
    if (base == "xgemm")          return "Xgemm";
    if (base == "xgemm_direct")   return "XgemmDirect";
    if (base == "xgemv")          return "Xgemv";
    if (base == "xgemv_fast")     return "XgemvFast";
    if (base == "xgemv_fast_rot") return "XgemvFastRot";
    if (base == "xger")           return "Xger";
    if (base == "xaxpy")          return "Xaxpy";
    if (base == "xdot")           return "Xdot";
    if (base == "invert")         return "Invert";
    return {};  // unknown — caller skips
}

void applyOne(cl_device_id dev, const std::string& path) {
    std::ifstream f(path);
    if (!f) return;
    std::stringstream buf; buf << f.rdbuf();
    auto blob = buf.str();

    auto family    = extractString(blob, "kernel_family");
    auto params    = extractString(blob, "best_parameters");
    auto precision = extractString(blob, "precision");
    if (family.empty() || params.empty() || precision.empty()) {
        std::fprintf(stderr,
            "chipBLAS: tuning %s missing kernel_family/best_parameters/precision; skipped\n",
            path.c_str());
        return;
    }
    auto kernel = familyToKey(family);
    if (kernel.empty()) {
        std::fprintf(stderr,
            "chipBLAS: tuning %s has unknown kernel_family '%s'; skipped\n",
            path.c_str(), family.c_str());
        return;
    }

    auto map = parseParameters(params);
    if (map.empty()) {
        std::fprintf(stderr,
            "chipBLAS: tuning %s parsed no parameters; skipped\n", path.c_str());
        return;
    }

    auto rc = clblast::OverrideParameters(dev, kernel,
                                          parsePrecision(precision), map);
    if (rc != clblast::StatusCode::kSuccess) {
        std::fprintf(stderr,
            "chipBLAS: OverrideParameters(%s) failed (status %d); skipped\n",
            path.c_str(), static_cast<int>(rc));
    }
}

} // namespace

void applyTuningOverrides(Handle& h) {
    if (!h.isOpenCL || !h.device) return;
    const char* dir = std::getenv("CHIPBLAS_TUNING_DIR");
    if (!dir || !dir[0]) return;

    DIR* dp = opendir(dir);
    if (!dp) {
        std::fprintf(stderr,
            "chipBLAS: CHIPBLAS_TUNING_DIR='%s' is not readable; skipped\n", dir);
        return;
    }
    int count = 0;
    while (auto* ent = readdir(dp)) {
        std::string name = ent->d_name;
        if (name.size() < 6 || name.substr(name.size() - 5) != ".json") continue;
        applyOne(h.device, std::string(dir) + "/" + name);
        ++count;
    }
    closedir(dp);
    if (std::getenv("CHIPBLAS_TRACE")) {
        std::fprintf(stderr,
            "[chipblas] applied tuning overrides from %s (%d files)\n",
            dir, count);
    }
}

} // namespace chipblas
