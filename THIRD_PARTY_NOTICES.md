# Third-party notices

binCV's own license is not yet chosen; until it is, no license is granted to the code in
this repository. Independently of that choice, the repository contains the third-party
material listed here, each with the notice its license requires.

| what | where | source and terms |
|---|---|---|
| the 256-pair ORB sampling table | `include/bincv/ops/orbPattern.hpp` | OpenCV `modules/features2d/src/orb.cpp`, BSD 3-clause (below) |
| a sort comparator, copied verbatim | `tests/test_corner.cpp` (`greaterThanPtr`) | OpenCV `modules/imgproc/src/featureselect.cpp`, Intel License Agreement (BSD-style, below) |
| two 752×480 camera frames | `tests/images/1403715887284058112.png`, its `_bin_normalized` derivative, and `benchmark/realframe.bin` (the same frame, raw) | the EuRoC MAV dataset, sequence `V1_02_medium`, camera `cam0` (below) |

The examples under `examples/` and two parameter choices in `include/bincv/ops/` cite
HybVIO as the design they follow (its keypoint-culling policy and its detection radius);
that is a description of a design choice, and no HybVIO code is included.

## EuRoC MAV dataset (the test and benchmark frames)

The frames named above are reproduced from the EuRoC micro aerial vehicle datasets,
published by the Autonomous Systems Lab, ETH Zürich, and used under the terms stated on
the dataset's ETH Research Collection record
(<https://doi.org/10.3929/ethz-b-000690084>). The dataset asks to be cited as:

> M. Burri, J. Nikolic, P. Gohl, T. Schneider, J. Rehder, S. Omari, M. W. Achtelik and
> R. Siegwart, *The EuRoC micro aerial vehicle datasets*, The International Journal of
> Robotics Research, 35(10), 2016.

The sequence-level results in `docs/reports/` were measured on the whole `V1_02_medium`
sequence, which is not included in the repository; `GETTING_STARTED.md` says how to point
the examples at it.

## OpenCV `bit_pattern_31_` (in `include/bincv/ops/orbPattern.hpp`)

The 256-pair ORB sampling table is copied verbatim from OpenCV,
`modules/features2d/src/orb.cpp`, whose file-level license is the BSD 3-clause
license below (note: not OpenCV's newer Apache-2.0 project license — the
file-level notice governs this material). The same notice is reproduced in the
header that contains the copied values.

```
Software License Agreement (BSD License)

Copyright (c) 2009, Willow Garage, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.
 * Neither the name of the Willow Garage nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

## OpenCV `greaterThanPtr` (in `tests/test_corner.cpp`)

The sort comparator that the corner-selection test uses as its reference is copied
verbatim from OpenCV, `modules/imgproc/src/featureselect.cpp`, so that the test's tie
order is OpenCV's own rather than a restatement of it. That file carries the Intel
License Agreement below.

```
                        Intel License Agreement
                For Open Source Computer Vision Library

 Copyright (C) 2000, Intel Corporation, all rights reserved.
 Third party copyrights are property of their respective owners.

 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

   * Redistribution's of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.

   * Redistribution's in binary form must reproduce the above copyright notice,
     this list of conditions and the following disclaimer in the documentation
     and/or other materials provided with the distribution.

   * The name of Intel Corporation may not be used to endorse or promote products
     derived from this software without specific prior written permission.

 This software is provided by the copyright holders and contributors "as is" and
 any express or implied warranties, including, but not limited to, the implied
 warranties of merchantability and fitness for a particular purpose are disclaimed.
 In no event shall the Intel Corporation or contributors be liable for any direct,
 indirect, incidental, special, exemplary, or consequential damages
 (including, but not limited to, procurement of substitute goods or services;
 loss of use, data, or profits; or business interruption) however caused
 and on any theory of liability, whether in contract, strict liability,
 or tort (including negligence or otherwise) arising in any way out of
 the use of this software, even if advised of the possibility of such damage.
```
