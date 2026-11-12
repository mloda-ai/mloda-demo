Three clients' books of an invented tax office, business year 2025.
The bookings are invented, since real client books are confidential. The formats are real.

- `client_a/EXTF_Buchungsstapel.csv`: DATEV-Format EXTF 700 booking batch, states SKR04.
- `client_c/`: the same bookings as a GDPdU dossier (`index.xml` and `GL.txt`).
- `client_b/EXTF_Buchungsstapel.csv`: bookings in SKR03; the header states no chart.
- `client_b_corrected/EXTF_Buchungsstapel.csv`: the same batch with SKR03 stated in the header.

Clients A and C are the `twin_2025` test books of the `german_ledger` package in
[mloda-registry](https://github.com/mloda-ai/mloda-registry) (Apache 2.0): client A with the client number and label changed, client C unchanged.
Client B is derived from client A.

The DATEV header and column line are taken from [ledermann/datev](https://github.com/ledermann/datev),
under the following license:

> The MIT License (MIT)
>
> Copyright (c) 2016-2026 Georg Ledermann and contributors
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in
> all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
> THE SOFTWARE.
