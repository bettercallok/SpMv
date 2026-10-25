// mmio.h — Matrix Market reader: coordinate real/integer/pattern, general or
// symmetric (expanded to full storage). Throws std::runtime_error on bad input.
#pragma once

#include <string>

#include "formats.h"

CsrMatrix read_matrix_market(const std::string& path);
