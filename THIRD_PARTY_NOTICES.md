# Third-party code notices

This file identifies the third-party code bundled with D-ripALM. The project's root MIT license does not replace the terms applicable to these files. Sources and license texts below were checked on November 19, 2026. The inspected upstream revisions are reference points, not assertions about the unrecorded original import revisions.

## IDEAL

- Local file: `solver/IDEAL-master/AccGDSolver.m`.
- Source: [IDEAL, Neurips_code/AccGDSolver.m](https://github.com/hongzhoulin89/IDEAL-Inexact-DEcentralized-Accelerated-Augmented-Lagrangian-Method/blob/9289c6dfd320f6b84230ae7cafe8cce091568618/Neurips_code/AccGDSolver.m).
- Local adaptations add configurable stopping tests, iteration and time limits, and result reporting for the D-ripALM experiments.
- License: MIT. The [upstream license](https://github.com/hongzhoulin89/IDEAL-Inexact-DEcentralized-Accelerated-Augmented-Lagrangian-Method/blob/9289c6dfd320f6b84230ae7cafe8cce091568618/LICENSE) is reproduced below.

```text
MIT License

Copyright (c) 2020 Hongzhou Lin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## NIDS / PG-EXTRA

- Local file: `solver/NIDS-master/fcns/NIDS.m`.
- Source: [NIDS, fcns/NIDS.m](https://github.com/mingyan08/NIDS/blob/ba3ba083de41aa72bbfb28c4c2bc2aa3d47926dd/fcns/NIDS.m).
- The source header names Zhi Li, Wei Shi, and Ming Yan. Their attribution is retained in the local file.
- Local adaptations add consensus and KKT stopping tests, result reporting, and changes to iteration-history storage.
- **License status unresolved:** no explicit license grant was found in the inspected upstream repository's file tree, README, or NIDS.m header at revision `ba3ba083de41aa72bbfb28c4c2bc2aa3d47926dd`. Public availability and attribution alone do not establish permission to redistribute modified code. This project does not license the upstream material under MIT.

Before distributing this adaptation, obtain and retain permission from the relevant rights holders, or replace/remove it from the distribution. This notice records the unresolved status; it does not provide that permission.

## readMNIST

- Local file: `utils/readMNIST.m`, whose header credits Siddharth Hegde.
- Source: Sid H, [Read digits and labels from MNIST database](https://www.mathworks.com/matlabcentral/fileexchange/27675-read-digits-and-labels-from-mnist-database), version 1.0.0, published May 20, 2010.
- The local version leaves the border-trimming call disabled, preserving the full 28-by-28 images, and normalizes pixel values by 255.
- License: BSD two-clause form. The original release's license, displayed by its “View License” control, is reproduced below; the copyright name is retained exactly as published.
- [Original license text](https://addons.mathworks.com/downloads/e5/e5711758-4a80-11e4-9553-005056977bd0/1.0.0/license.txt).

```text
Copyright (c) 2010, Sid H
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
