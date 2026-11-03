# Serein backtest (dev) - 2026-11-02 11:44
code version: 49a94cb + uncommitted changes
rows: train 130050, val 32300, conf 32300, test 32725; leads [0, 1, 2, 3]
graph: 425 nodes, 4250 edges
features: 64

## training smokecast (seed 0)
  epoch   0 train 0.1617 val 0.1192 *
  epoch   1 train 0.1401 val 0.1110 *
  epoch   3 train 0.1318 val 0.1060 *
  epoch   4 train 0.1263 val 0.1046 *
  epoch   6 train 0.1259 val 0.1043 *
  epoch   7 train 0.1218 val 0.1030 *
  epoch  10 train 0.1178 val 0.1011 *
  epoch  11 train 0.1170 val 0.1000 *
  epoch  18 train 0.1115 val 0.0997 *
  epoch  20 train 0.1122 val 0.1006
  epoch  23 train 0.1117 val 0.0988 *
  epoch  29 train 0.1087 val 0.0987 *
  epoch  30 train 0.1089 val 0.1011
  epoch  40 train 0.1027 val 0.1016
  epoch  50 train 0.0999 val 0.1044
  early stop at epoch 59; best val 0.0987

## training smokecast (seed 1)
  epoch   0 train 0.1646 val 0.1173 *
  epoch   1 train 0.1466 val 0.1120 *
  epoch   2 train 0.1364 val 0.1085 *
  epoch   3 train 0.1312 val 0.1067 *
  epoch   4 train 0.1293 val 0.1055 *
  epoch   7 train 0.1223 val 0.1018 *
  epoch  10 train 0.1182 val 0.1009 *
  epoch  11 train 0.1175 val 0.1006 *
  epoch  13 train 0.1144 val 0.1005 *
  epoch  16 train 0.1142 val 0.0988 *
  epoch  20 train 0.1111 val 0.1002
  epoch  30 train 0.1065 val 0.1019
  epoch  40 train 0.1050 val 0.1017
  early stop at epoch 46; best val 0.0988

## training smokecast (seed 2)
  epoch   0 train 0.1626 val 0.1160 *
  epoch   1 train 0.1450 val 0.1135 *
  epoch   2 train 0.1355 val 0.1100 *
  epoch   3 train 0.1328 val 0.1074 *
  epoch   4 train 0.1294 val 0.1059 *
  epoch   5 train 0.1259 val 0.1052 *
  epoch   6 train 0.1258 val 0.1048 *
  epoch   7 train 0.1222 val 0.1039 *
  epoch   8 train 0.1202 val 0.1024 *
  epoch   9 train 0.1194 val 0.1009 *
  epoch  10 train 0.1214 val 0.1039
  epoch  15 train 0.1162 val 0.0996 *
  epoch  19 train 0.1146 val 0.0993 *
  epoch  20 train 0.1124 val 0.0993 *
  epoch  30 train 0.1059 val 0.1038
  epoch  32 train 0.1057 val 0.0985 *
  epoch  40 train 0.1028 val 0.1004
  epoch  50 train 0.0988 val 0.1028
  epoch  60 train 0.0962 val 0.1040
  early stop at epoch 62; best val 0.0985

## training smokecast (seed 3)
  epoch   0 train 0.1639 val 0.1175 *
  epoch   1 train 0.1404 val 0.1103 *
  epoch   2 train 0.1376 val 0.1090 *
  epoch   4 train 0.1276 val 0.1039 *
  epoch   8 train 0.1205 val 0.1034 *
  epoch   9 train 0.1201 val 0.1021 *
  epoch  10 train 0.1185 val 0.1035
  epoch  12 train 0.1161 val 0.1003 *
  epoch  20 train 0.1106 val 0.0996 *
  epoch  28 train 0.1083 val 0.0994 *
  epoch  30 train 0.1070 val 0.1005
  epoch  40 train 0.1024 val 0.1017
  epoch  50 train 0.1005 val 0.1026
  early stop at epoch 58; best val 0.0994

## training smokecast (seed 4)
  epoch   0 train 0.1625 val 0.1162 *
  epoch   1 train 0.1415 val 0.1103 *
  epoch   2 train 0.1358 val 0.1077 *
  epoch   4 train 0.1276 val 0.1032 *
  epoch  10 train 0.1191 val 0.1032 *
  epoch  11 train 0.1190 val 0.1030 *
  epoch  12 train 0.1173 val 0.1009 *
  epoch  13 train 0.1148 val 0.1000 *
  epoch  17 train 0.1153 val 0.0994 *
  epoch  20 train 0.1120 val 0.0993 *
  epoch  21 train 0.1101 val 0.0992 *
  epoch  27 train 0.1080 val 0.0990 *
  epoch  30 train 0.1051 val 0.1016
  epoch  40 train 0.1022 val 0.0984 *
  epoch  50 train 0.0984 val 0.1013
  epoch  60 train 0.0958 val 0.1032
  epoch  70 train 0.0941 val 0.1023
  early stop at epoch 70; best val 0.0984

## training smokecast_nograph (seed 0)
  epoch   0 train 0.1709 val 0.1182 *
  epoch   1 train 0.1412 val 0.1115 *
  epoch   2 train 0.1377 val 0.1089 *
  epoch   3 train 0.1338 val 0.1061 *
  epoch   4 train 0.1288 val 0.1046 *
  epoch   6 train 0.1301 val 0.1036 *
  epoch   7 train 0.1252 val 0.1028 *
  epoch  10 train 0.1223 val 0.1018 *
  epoch  13 train 0.1204 val 0.1015 *
  epoch  18 train 0.1188 val 0.1008 *
  epoch  20 train 0.1197 val 0.1022
  epoch  23 train 0.1205 val 0.1002 *
  epoch  27 train 0.1171 val 0.1000 *
  epoch  30 train 0.1159 val 0.1002
  epoch  36 train 0.1140 val 0.0995 *
  epoch  40 train 0.1140 val 0.1009
  epoch  50 train 0.1136 val 0.1015
  epoch  60 train 0.1123 val 0.1007
  epoch  65 train 0.1112 val 0.0994 *
  epoch  70 train 0.1104 val 0.1006
  epoch  80 train 0.1091 val 0.1004
  epoch  90 train 0.1103 val 0.1017
  early stop at epoch 95; best val 0.0994

## training gbm
warning cut-offs P(>=35.5) tuned on calibration: {0: 0.5750000000000002, 1: 0.5250000000000001, 2: 0.47500000000000014, 3: 0.4250000000000001}

## Results: all site-days
RMSE (µg/m³, lower is better)
lead                   0      1      2      3
model                                        
aqm_bc             10.14  11.02  12.35    NaN
climatology        13.44  13.76  13.81  13.71
gbm                 5.48   8.87  10.98  12.56
hrrr_smoke         12.63   9.05    NaN    NaN
noaa_best          10.14  11.02  12.35  15.48
persistence        10.15  13.79  14.97  15.48
smokecast           4.46   7.86   9.70  11.38
smokecast_nograph   4.76   8.15  10.27  11.98

RMSE on smoke-affected days (HMS plume overhead yesterday)
lead                   0      1      2      3
model                                        
aqm_bc             14.77  15.87  17.50    NaN
climatology        19.95  20.12  19.77  19.70
gbm                 8.09  12.85  15.70  18.22
hrrr_smoke         18.75  12.73    NaN    NaN
noaa_best          14.77  15.87  17.50  22.55
persistence        15.03  20.29  21.66  22.55
smokecast           6.47  11.17  13.83  16.43
smokecast_nograph   6.96  11.67  14.69  17.29

USG+ critical success index (higher is better)
lead                    0      1      2      3
model                                         
aqm_bc              0.369  0.285  0.142    NaN
climatology         0.000  0.000  0.000  0.000
gbm                 0.646  0.462  0.243  0.080
hrrr_smoke          0.509  0.468    NaN    NaN
noaa_best           0.369  0.285  0.142  0.151
persistence         0.376  0.257  0.179  0.151
smokecast           0.654  0.534  0.449  0.310
smokecast_nograph   0.677  0.527  0.403  0.253
smokecast_warnrule  0.660  0.536  0.446  0.306

90% interval coverage (target 0.90)
lead                   0      1      2      3
model                                        
gbm                0.911  0.908  0.899  0.907
smokecast          0.903  0.891  0.898  0.906
smokecast_nograph  0.903  0.893  0.902  0.912

Head-to-head (same rows; skill = 1 - RMSE_smokecast/RMSE_baseline; CI = 95% day-bootstrap of RMSE diff)
 lead                vs     n  rmse_smokecast  rmse_baseline  skill   ci_lo  ci_hi
    0         noaa_best 25030           4.460         10.138  0.560  -8.284 -3.215
    0            aqm_bc 25030           4.460         10.138  0.560  -8.284 -3.215
    0        hrrr_smoke 25030           4.460         12.630  0.647 -12.846 -3.957
    0       persistence 24732           4.477         10.149  0.559  -8.435 -2.640
    0               gbm 25030           4.460          5.476  0.186  -1.945 -0.142
    0 smokecast_nograph 25030           4.460          4.759  0.063  -0.667  0.051
    1         noaa_best 25041           7.860         11.022  0.287  -4.720 -1.521
    1            aqm_bc 25041           7.860         11.022  0.287  -4.720 -1.521
    1        hrrr_smoke 25041           7.860          9.053  0.132  -2.399 -0.197
    1       persistence 24562           7.813         13.795  0.434  -9.349 -2.667
    1               gbm 25041           7.860          8.873  0.114  -1.999  0.158
    1 smokecast_nograph 25041           7.860          8.146  0.035  -0.677  0.153
    2         noaa_best 24730           9.704         12.348  0.214  -3.889 -1.566
    2            aqm_bc 24730           9.704         12.348  0.214  -3.889 -1.566
    2       persistence 24144           9.470         14.965  0.367  -7.846 -3.168
    2               gbm 24730           9.704         10.980  0.116  -2.179 -0.106
    2 smokecast_nograph 24730           9.704         10.272  0.055  -0.922 -0.180
    3         noaa_best 23787          11.179         15.475  0.278  -7.158 -1.946
    3       persistence 23787          11.179         15.475  0.278  -7.158 -1.946
    3               gbm 24432          11.384         12.564  0.094  -2.311  0.039
    3 smokecast_nograph 24432          11.384         11.980  0.050  -1.081 -0.085

vs. official EPA AQI forecast (reporting areas, category scores)
 lead          who  n_area_days  cat_acc  usg_events  usg_pod  usg_far  usg_csi
    0    smokecast        10948    0.926         224    0.777    0.179    0.664
    0 epa_forecast        10948    0.842         224    0.665    0.402    0.460
    1    smokecast        10775    0.879         244    0.709    0.319    0.532
    1 epa_forecast        10775    0.827         244    0.459    0.467    0.327
    2    smokecast         6631    0.893         190    0.658    0.302    0.512
    2 epa_forecast         6631    0.855         190    0.368    0.340    0.310
    3    smokecast         5576    0.888         118    0.432    0.514    0.297
    3 epa_forecast         5576    0.885         118    0.110    0.316    0.105