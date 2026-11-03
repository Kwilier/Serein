# Serein backtest (dev) - 2026-11-02 12:03
code version: 49a94cb + uncommitted changes
rows: train 130050, val 32300, conf 32300, test 32725; leads [0, 1, 2, 3]
graph: 425 nodes, 4250 edges
features: 64
leave-sites-out: 85 of 425 monitors hidden

## training smokecast (seed 0)
  epoch   0 train 0.1642 val 0.1252 *
  epoch   1 train 0.1424 val 0.1171 *
  epoch   3 train 0.1342 val 0.1110 *
  epoch   4 train 0.1282 val 0.1098 *
  epoch   6 train 0.1284 val 0.1090 *
  epoch   7 train 0.1238 val 0.1074 *
  epoch  10 train 0.1194 val 0.1064 *
  epoch  11 train 0.1190 val 0.1053 *
  epoch  13 train 0.1165 val 0.1047 *
  epoch  18 train 0.1130 val 0.1046 *
  epoch  20 train 0.1148 val 0.1061
  epoch  23 train 0.1140 val 0.1040 *
  epoch  24 train 0.1117 val 0.1038 *
  epoch  30 train 0.1113 val 0.1062
  epoch  36 train 0.1052 val 0.1038 *
  epoch  40 train 0.1042 val 0.1084
  epoch  50 train 0.1011 val 0.1080
  epoch  60 train 0.0995 val 0.1067
  early stop at epoch 66; best val 0.1038

## training smokecast (seed 1)
  epoch   0 train 0.1665 val 0.1229 *
  epoch   1 train 0.1487 val 0.1164 *
  epoch   2 train 0.1383 val 0.1134 *
  epoch   3 train 0.1333 val 0.1118 *
  epoch   4 train 0.1308 val 0.1093 *
  epoch   7 train 0.1244 val 0.1064 *
  epoch   8 train 0.1239 val 0.1059 *
  epoch  10 train 0.1202 val 0.1051 *
  epoch  11 train 0.1194 val 0.1049 *
  epoch  15 train 0.1165 val 0.1049 *
  epoch  16 train 0.1158 val 0.1041 *
  epoch  20 train 0.1127 val 0.1045
  epoch  27 train 0.1099 val 0.1040 *
  epoch  30 train 0.1085 val 0.1069
  epoch  40 train 0.1084 val 0.1047
  epoch  50 train 0.1006 val 0.1077
  early stop at epoch 57; best val 0.1040

## training smokecast (seed 2)
  epoch   0 train 0.1647 val 0.1230 *
  epoch   1 train 0.1467 val 0.1183 *
  epoch   2 train 0.1374 val 0.1154 *
  epoch   3 train 0.1348 val 0.1120 *
  epoch   4 train 0.1318 val 0.1109 *
  epoch   5 train 0.1281 val 0.1098 *
  epoch   6 train 0.1278 val 0.1091 *
  epoch   8 train 0.1221 val 0.1059 *
  epoch  10 train 0.1231 val 0.1081
  epoch  13 train 0.1165 val 0.1058 *
  epoch  15 train 0.1180 val 0.1043 *
  epoch  20 train 0.1143 val 0.1037 *
  epoch  30 train 0.1073 val 0.1071
  epoch  32 train 0.1064 val 0.1036 *
  epoch  40 train 0.1054 val 0.1039
  epoch  50 train 0.0999 val 0.1069
  epoch  60 train 0.0974 val 0.1059
  early stop at epoch 62; best val 0.1036

## training smokecast (seed 3)
  epoch   0 train 0.1662 val 0.1230 *
  epoch   1 train 0.1421 val 0.1160 *
  epoch   2 train 0.1398 val 0.1149 *
  epoch   3 train 0.1367 val 0.1140 *
  epoch   4 train 0.1300 val 0.1094 *
  epoch   8 train 0.1229 val 0.1084 *
  epoch   9 train 0.1223 val 0.1068 *
  epoch  10 train 0.1211 val 0.1083
  epoch  12 train 0.1177 val 0.1048 *
  epoch  20 train 0.1125 val 0.1043 *
  epoch  30 train 0.1091 val 0.1058
  epoch  31 train 0.1078 val 0.1033 *
  epoch  40 train 0.1035 val 0.1073
  epoch  50 train 0.1018 val 0.1066
  epoch  60 train 0.0974 val 0.1085
  early stop at epoch 61; best val 0.1033

## training smokecast (seed 4)
  epoch   0 train 0.1644 val 0.1221 *
  epoch   1 train 0.1434 val 0.1163 *
  epoch   2 train 0.1378 val 0.1143 *
  epoch   4 train 0.1296 val 0.1079 *
  epoch  10 train 0.1208 val 0.1075 *
  epoch  11 train 0.1211 val 0.1075 *
  epoch  12 train 0.1189 val 0.1053 *
  epoch  13 train 0.1171 val 0.1044 *
  epoch  16 train 0.1162 val 0.1043 *
  epoch  17 train 0.1171 val 0.1038 *
  epoch  20 train 0.1135 val 0.1037 *
  epoch  26 train 0.1112 val 0.1036 *
  epoch  30 train 0.1076 val 0.1054
  epoch  31 train 0.1081 val 0.1036 *
  epoch  40 train 0.1040 val 0.1033 *
  epoch  50 train 0.1000 val 0.1045
  epoch  60 train 0.0983 val 0.1058
  epoch  70 train 0.0955 val 0.1064
  early stop at epoch 70; best val 0.1033

## training smokecast_nograph (seed 0)
  epoch   0 train 0.1734 val 0.1234 *
  epoch   1 train 0.1430 val 0.1181 *
  epoch   2 train 0.1397 val 0.1150 *
  epoch   3 train 0.1357 val 0.1127 *
  epoch   4 train 0.1306 val 0.1113 *
  epoch   6 train 0.1318 val 0.1103 *
  epoch   7 train 0.1270 val 0.1094 *
  epoch  10 train 0.1239 val 0.1085 *
  epoch  13 train 0.1220 val 0.1079 *
  epoch  16 train 0.1215 val 0.1078 *
  epoch  18 train 0.1203 val 0.1072 *
  epoch  20 train 0.1212 val 0.1081
  epoch  23 train 0.1226 val 0.1069 *
  epoch  30 train 0.1174 val 0.1068 *
  epoch  36 train 0.1155 val 0.1060 *
  epoch  40 train 0.1157 val 0.1076
  epoch  46 train 0.1143 val 0.1058 *
  epoch  50 train 0.1147 val 0.1074
  epoch  60 train 0.1137 val 0.1072
  epoch  65 train 0.1128 val 0.1057 *
  epoch  70 train 0.1119 val 0.1068
  epoch  80 train 0.1107 val 0.1062
  epoch  90 train 0.1115 val 0.1076
  early stop at epoch 95; best val 0.1057
warning cut-offs P(>=35.5) tuned on calibration: {0: 0.5000000000000002, 1: 0.4500000000000001, 2: 0.4250000000000001, 3: 0.22500000000000003}

## Results: all site-days
RMSE (µg/m³, lower is better)
lead                   0      1      2      3
model                                        
aqm_bc             10.14  11.02  12.35    NaN
climatology        13.44  13.76  13.81  13.71
hrrr_smoke         12.63   9.05    NaN    NaN
noaa_best          10.14  11.02  12.35  15.48
persistence        10.15  13.79  14.97  15.48
smokecast           5.08   7.98   9.75  11.30
smokecast_nograph   5.58   8.13  10.25  12.40

RMSE on smoke-affected days (HMS plume overhead yesterday)
lead                   0      1      2      3
model                                        
aqm_bc             14.77  15.87  17.50    NaN
climatology        19.95  20.12  19.77  19.70
hrrr_smoke         18.75  12.73    NaN    NaN
noaa_best          14.77  15.87  17.50  22.55
persistence        15.03  20.29  21.66  22.55
smokecast           7.39  11.33  13.89  16.28
smokecast_nograph   8.16  11.60  14.62  17.85

RMSE at HIDDEN monitors (no readings; persistence/climatology use data a real unmonitored place would not have)
lead                   0      1      2      3
model                                        
aqm_bc             10.46  11.14  11.93    NaN
climatology        12.66  12.66  12.25  11.90
hrrr_smoke          9.12  10.10    NaN    NaN
noaa_best          10.46  11.14  11.93  14.10
persistence        10.00  13.45  13.46  14.10
smokecast           6.98   8.49  10.05  11.31
smokecast_nograph   8.42   9.28  10.88  13.53

USG+ critical success index (higher is better)
lead                    0      1      2      3
model                                         
aqm_bc              0.369  0.285  0.142    NaN
climatology         0.000  0.000  0.000  0.000
hrrr_smoke          0.509  0.468    NaN    NaN
noaa_best           0.369  0.285  0.142  0.151
persistence         0.376  0.257  0.179  0.151
smokecast           0.640  0.508  0.442  0.312
smokecast_nograph   0.663  0.528  0.386  0.244
smokecast_warnrule  0.640  0.498  0.426  0.252

90% interval coverage (target 0.90)
lead                   0      1      2      3
model                                        
smokecast          0.904  0.894  0.899  0.907
smokecast_nograph  0.901  0.894  0.901  0.911

Head-to-head (same rows; skill = 1 - RMSE_smokecast/RMSE_baseline; CI = 95% day-bootstrap of RMSE diff)
 lead                vs     n  rmse_smokecast  rmse_baseline  skill   ci_lo  ci_hi
    0         noaa_best 25030           5.084         10.138  0.499  -7.316 -2.919
    0            aqm_bc 25030           5.084         10.138  0.499  -7.316 -2.919
    0        hrrr_smoke 25030           5.084         12.630  0.597 -12.253 -3.383
    0       persistence 24732           5.101         10.149  0.497  -7.617 -2.274
    0 smokecast_nograph 25030           5.084          5.578  0.089  -0.941 -0.107
    1         noaa_best 25041           7.979         11.022  0.276  -4.416 -1.573
    1            aqm_bc 25041           7.979         11.022  0.276  -4.416 -1.573
    1        hrrr_smoke 25041           7.979          9.053  0.119  -2.136 -0.126
    1       persistence 24562           7.939         13.795  0.424  -9.010 -2.721
    1 smokecast_nograph 25041           7.979          8.133  0.019  -0.458  0.121
    2         noaa_best 24730           9.749         12.348  0.210  -3.623 -1.605
    2            aqm_bc 24730           9.749         12.348  0.210  -3.623 -1.605
    2       persistence 24144           9.548         14.965  0.362  -7.678 -3.245
    2 smokecast_nograph 24730           9.749         10.249  0.049  -1.133 -0.052
    3         noaa_best 23787          11.151         15.475  0.279  -7.114 -1.976
    3       persistence 23787          11.151         15.475  0.279  -7.114 -1.976
    3 smokecast_nograph 24432          11.297         12.399  0.089  -1.833 -0.377

vs. official EPA AQI forecast (reporting areas, category scores)
 lead          who  n_area_days  cat_acc  usg_events  usg_pod  usg_far  usg_csi
    0    smokecast        10948    0.919         224    0.754    0.172    0.653
    0 epa_forecast        10948    0.842         224    0.665    0.402    0.460
    1    smokecast        10775    0.882         244    0.713    0.365    0.506
    1 epa_forecast        10775    0.827         244    0.459    0.467    0.327
    2    smokecast         6631    0.894         190    0.674    0.326    0.508
    2 epa_forecast         6631    0.855         190    0.368    0.340    0.310
    3    smokecast         5576    0.893         118    0.508    0.615    0.280
    3 epa_forecast         5576    0.885         118    0.110    0.316    0.105