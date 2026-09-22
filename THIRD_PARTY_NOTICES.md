# Third-party notices

AdsDrift contains adapted or consolidated operator code from these projects:

- [EquiformerV3](https://github.com/atomicarchitects/equiformer_v3), MIT License.
- [FAIR Chemistry](https://github.com/facebookresearch/fairchem), MIT License,
  Copyright (c) Meta Platforms, Inc. and affiliates.
- [e3nn](https://github.com/e3nn/e3nn), MIT License, including the Wigner-D
  implementation identified in `model/generator/core/geometry.py`.

The above components remain subject to their upstream licenses. Their MIT
license text follows.

```text
MIT License

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

## External model and data licenses

No third-party model weights or training data are distributed in this
repository. AdsDrift's training objective can use MACE-MH-1, but users must
obtain it separately and comply with its upstream ASL license. OC20/OC20-Dense
data must likewise be obtained from its official source under its own terms.
