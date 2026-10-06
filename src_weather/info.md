# Weather module notes

The implementation and resampling assumptions are documented in
[weather.md](./weather.md). In short, the 15-minute values are estimates from
hourly source data; original hourly values are retained, continuous readings
are interpolated between valid adjacent hours, and precipitation/sunshine
totals are evenly allocated so their hourly sums are conserved.
