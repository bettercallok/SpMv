#include "mmio.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

CsrMatrix read_matrix_market(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);

  std::string line, banner, object, format, field, symmetry;
  std::getline(in, line);
  std::istringstream header(line);
  header >> banner >> object >> format >> field >> symmetry;
  if (banner != "%%MatrixMarket" || object != "matrix" || format != "coordinate")
    throw std::runtime_error(path + ": only sparse coordinate Matrix Market files are supported");
  const bool pattern = field == "pattern";
  const bool symmetric = symmetry == "symmetric";
  if (field == "complex" || (!symmetric && symmetry != "general"))
    throw std::runtime_error(path + ": unsupported field/symmetry " + field + " " + symmetry);

  while (std::getline(in, line) && (line.empty() || line[0] == '%')) {
  }
  long long rows = 0, cols = 0, entries = 0;
  std::istringstream size_line(line);
  if (!(size_line >> rows >> cols >> entries)) throw std::runtime_error(path + ": bad size line");

  std::vector<int> ri, ci;
  std::vector<float> v;
  for (long long e = 0; e < entries; ++e) {
    long long r, c;
    double val = 1.0;
    if (!(in >> r >> c) || (!pattern && !(in >> val)))
      throw std::runtime_error(path + ": unexpected end of file");
    if (r < 1 || r > rows || c < 1 || c > cols) throw std::runtime_error(path + ": index out of range");
    ri.push_back(static_cast<int>(r - 1));
    ci.push_back(static_cast<int>(c - 1));
    v.push_back(static_cast<float>(val));
    if (symmetric && r != c) {  // mirror the stored lower triangle
      ri.push_back(static_cast<int>(c - 1));
      ci.push_back(static_cast<int>(r - 1));
      v.push_back(static_cast<float>(val));
    }
  }
  return coo_to_csr(static_cast<int>(rows), static_cast<int>(cols), ri, ci, v);
}
