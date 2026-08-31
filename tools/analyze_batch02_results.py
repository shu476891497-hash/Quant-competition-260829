"""Parse the frozen QuantConnect batch-02 runtime statistics and apply BH FDR."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

STRUCTURE = {
    "ES_FOID5": "H1:N4070/I-0.007/R-0.006/T-0.42/P0.6723/Q8;H5:N3806/I-0.016/R-0.029/T-0.70/P0.4843/Q4;H21:N2750/I0.025/R0.013/T0.69/P0.4925/Q22",
    "ES_NCURV": "H1:N4075/I-0.001/R-0.017/T-0.05/P0.9589/Q-1;H5:N3811/I0.020/R0.005/T0.95/P0.3418/Q10;H21:N2755/I-0.002/R0.015/T-0.07/P0.9431/Q-8",
    "ES_OIMAT": "H1:N4075/I0.017/R0.010/T1.04/P0.2967/Q0;H5:N3811/I0.025/R0.008/T0.88/P0.3792/Q-0;H21:N2755/I0.081/R0.065/T1.43/P0.1532/Q61",
    "ES_OIMD5": "H1:N4070/I-0.003/R-0.005/T-0.18/P0.8553/Q-7;H5:N3806/I-0.022/R-0.011/T-0.96/P0.3390/Q-12;H21:N2750/I0.013/R0.018/T0.35/P0.7244/Q30",
    "ES_OIRAT": "H1:N4075/I-0.014/R-0.003/T-0.86/P0.3909/Q-3;H5:N3811/I-0.027/R-0.003/T-0.97/P0.3336/Q-29;H21:N2755/I-0.066/R-0.030/T-1.17/P0.2410/Q-85",
    "ES_VMOI": "H1:N4075/I0.005/R-0.005/T0.34/P0.7355/Q2;H5:N3811/I0.023/R0.005/T0.91/P0.3605/Q25;H21:N2755/I0.045/R0.056/T1.00/P0.3155/Q81",
    "NQ_FOID5": "H1:N3588/I-0.020/R-0.022/T-1.01/P0.3122/Q-4;H5:N3344/I-0.043/R-0.050/T-1.45/P0.1457/Q-21;H21:N2368/I0.021/R0.010/T0.45/P0.6519/Q76",
    "NQ_NCURV": "H1:N3593/I-0.003/R0.014/T-0.15/P0.8786/Q1;H5:N3349/I0.062/R0.077/T2.69/P0.0071/Q55;H21:N2373/I0.066/R0.061/T1.99/P0.0463/Q103",
    "NQ_OIMAT": "H1:N3593/I0.004/R-0.005/T0.25/P0.8050/Q-4;H5:N3349/I0.008/R-0.005/T0.29/P0.7700/Q-9;H21:N2373/I0.044/R0.034/T0.72/P0.4722/Q56",
    "NQ_OIMD5": "H1:N3588/I-0.021/R-0.013/T-1.38/P0.1684/Q-6;H5:N3344/I-0.040/R-0.007/T-1.55/P0.1219/Q-15;H21:N2368/I0.017/R-0.010/T0.44/P0.6614/Q-13",
    "NQ_OIRAT": "H1:N3593/I0.002/R0.013/T0.13/P0.8975/Q0;H5:N3349/I0.001/R0.007/T0.02/P0.9860/Q8;H21:N2373/I-0.008/R-0.005/T-0.14/P0.8903/Q-4",
    "NQ_VMOI": "H1:N3593/I0.009/R-0.017/T0.38/P0.7024/Q-2;H5:N3349/I0.002/R-0.010/T0.06/P0.9491/Q-1;H21:N2373/I-0.018/R-0.001/T-0.38/P0.7075/Q1",
}

CFTC = {
    "ES_ALDIV": "H5:N769/I-0.055/R-0.053/T-1.38/P0.1662/Q-23;H21:N577/I-0.134/R-0.145/T-2.03/P0.0420/Q-135",
    "ES_ASPR": "H5:N769/I0.026/R0.040/T0.67/P0.5003/Q35;H21:N577/I0.056/R0.093/T0.73/P0.4680/Q109",
    "ES_DAS1": "H5:N768/I-0.009/R-0.009/T-0.16/P0.8719/Q-13;H21:N576/I0.029/R0.004/T0.76/P0.4457/Q4",
    "ES_DLS1": "H5:N768/I-0.001/R-0.044/T-0.03/P0.9785/Q-26;H21:N576/I-0.032/R-0.002/T-1.01/P0.3105/Q16",
    "ES_DSPR": "H5:N769/I-0.027/R-0.020/T-0.58/P0.5618/Q-4;H21:N577/I0.067/R0.077/T1.19/P0.2322/Q122",
    "ES_LSPR": "H5:N769/I-0.007/R-0.032/T-0.16/P0.8702/Q-17;H21:N577/I0.044/R-0.001/T0.86/P0.3915/Q58",
    "NQ_ALDIV": "H5:N720/I-0.026/R-0.006/T-0.66/P0.5093/Q-21;H21:N540/I-0.044/R-0.031/T-0.72/P0.4704/Q-44",
    "NQ_ASPR": "H5:N720/I-0.009/R0.029/T-0.26/P0.7936/Q19;H21:N540/I0.037/R0.092/T0.97/P0.3323/Q159",
    "NQ_DAS1": "H5:N719/I-0.002/R-0.046/T-0.06/P0.9524/Q-35;H21:N539/I-0.015/R-0.034/T-0.58/P0.5589/Q1",
    "NQ_DLS1": "H5:N719/I0.008/R0.066/T0.18/P0.8556/Q67;H21:N539/I0.003/R0.037/T0.09/P0.9311/Q68",
    "NQ_DSPR": "H5:N720/I0.002/R0.012/T0.06/P0.9523/Q35;H21:N540/I0.019/R0.076/T0.49/P0.6259/Q148",
    "NQ_LSPR": "H5:N720/I0.003/R0.037/T0.09/P0.9272/Q46;H21:N540/I0.016/R0.014/T0.32/P0.7467/Q51",
}

PATTERN = re.compile(
    r"H(?P<horizon>\d+):N(?P<n>\d+)/I(?P<ic>-?[\d.]+)"
    r"/R(?P<rank_ic>-?[\d.]+)/T(?P<nw_t>-?[\d.]+)"
    r"/P(?P<p_value>[\d.]+)/Q(?P<spread_bps>-?[\d.]+)"
)


def parse(family: str, values: dict[str, str]) -> pd.DataFrame:
    rows = []
    for key, payload in values.items():
        symbol, factor = key.split("_", 1)
        for part in payload.split(";"):
            match = PATTERN.fullmatch(part)
            if match is None:
                raise ValueError(f"cannot parse {key}: {part}")
            row = {name: float(value) for name, value in match.groupdict().items()}
            row.update({"family": family, "symbol": symbol, "factor": factor})
            row["horizon"] = int(row["horizon"])
            row["n"] = int(row["n"])
            rows.append(row)
    frame = pd.DataFrame(rows)
    order = frame["p_value"].sort_values().index
    m = len(frame)
    raw = frame.loc[order, "p_value"].to_numpy() * m / range(1, m + 1)
    adjusted = pd.Series(raw[::-1]).cummin().to_numpy()[::-1].clip(max=1.0)
    frame["fdr_q"] = 1.0
    frame.loc[order, "fdr_q"] = adjusted
    return frame.sort_values(["fdr_q", "p_value", "symbol", "factor", "horizon"])


def main() -> None:
    output = Path(__file__).parents[1] / "results"
    output.mkdir(exist_ok=True)
    for filename, family, values in (
        ("batch02_structure_oi.csv", "structure_oi", STRUCTURE),
        ("batch02_cftc_tff.csv", "cftc_tff", CFTC),
    ):
        frame = parse(family, values)
        frame.to_csv(output / filename, index=False)
        print(filename)
        print(frame.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
