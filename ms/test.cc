#include <hdf5.h>

#include "Table.h"
#include "IndexTree.h"

using namespace legms;
using namespace legms::ms;

IndexTreeL shape =
  IndexTreeL(
    {{0, 10, IndexTreeL(
          {{0, 5, IndexTreeL(4)}})},
     {10, 5, IndexTreeL(
         {{5, 5, IndexTreeL(8)}})}});
int
main(int argc, char* argv[]) {

  

}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// coding: utf-8
// End:
