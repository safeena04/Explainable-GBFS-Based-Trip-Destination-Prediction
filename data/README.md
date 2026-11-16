# Data

`raw/` holds the 7 daily CSV files (2024-03-01 .. 2024-03-07, Heidelberg) copied unchanged
from the original repository. One row = one COMPLETE trip (rental + return already paired
by vehicle_id). The pipeline hides vehicle_id from the candidate generator and uses it only
to know the true destination for evaluation.

Columns: vehicle_id, timestamp_lend, lng_lend, lat_lend, timestamp_returned, lng_returned,
lat_returned, current_range_meters_lend, pedelec_battery_lend, pedelec_battery_returned,
current_range_meters_returned.

Usage restriction (from the original README): the dataset is provided for reproducing the
experiments of Kerger and Stuckenschmidt. Any other use needs the authors' permission.

`processed/` is for files our own code generates. Never edit `raw/` by hand.
