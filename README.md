# When does a low-resolution image determine its blur kernel?

Blind super-resolution methods assume that from a downscaled image you can figure out
which blur kernel made it. We checked when that is true, and at a white source spectrum it
isn't. We build pairs of normal blur kernels (17x17, positive, sum to one) that are different
but give exactly the same low-resolution statistics, and a classifier can't tell them apart
even though it separates unrelated kernels easily. If we tilt the source spectrum a bit the
same classifier separates the same pairs almost perfectly.

## Running

```
pip install -r requirements.txt
cd code
python3 make_pairs.py
python3 run_experiment.py
python3 analyze.py
python3 fiber_rank.py
```

`fiber_rank.py --exact` also computes the rank over the rationals.

## Details

`make_pairs.py` builds `k_i = (G * v_i) / 2^16`, G on the even sub-lattice and v_2 is v_1
with two taps swapped (section 4 of the report). We check with exact fractions that the
autocorrelations differ but not on any even-even lag. Also 10 control pairs and 10 pairs
with the same kernel twice.

`run_experiment.py` draws Gaussian images on a 128x128 torus, blurs, keeps every second
pixel, and trains a gradient boosted classifier on log-periodogram features to tell the two
kernels apart. 200 and 800 images per kernel, white source and S = 1 + 0.8 cos(nu_2).

`fiber_rank.py` computes the rank of the derivative of kernel -> low-res power spectrum on
the 288-dimensional tangent space. 288 minus the rank is the number of invisible directions.

## Results

| arm | source | images per kernel | pairs | accuracy | 95% CI |
|---|---|---|---|---|---|
| matched | white | 200 | 20 | 0.508 | [0.493, 0.523] |
| matched | white | 800 | 20 | 0.495 | [0.485, 0.505] |
| matched | tilted | 200 | 20 | 0.983 | [0.975, 0.991] |
| matched | tilted | 800 | 20 | 0.992 | [0.988, 0.995] |
| same kernel | white | 200 | 10 | 0.508 | [0.480, 0.535] |
| different kernels | white | 200 | 10 | 0.996 | [0.992, 1.000] |

Rank of the Jacobian (tangent space has dimension 288):

| source | white | q=1 | q=2 | q=3 | q=4 |
|---|---|---|---|---|---|
| rank | 145 | 177 | 233 | 253 | 288 |
| invisible directions | 143 | 111 | 55 | 35 | 0 |
