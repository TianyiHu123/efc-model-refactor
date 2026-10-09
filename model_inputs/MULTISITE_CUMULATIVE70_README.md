# Depth-resolved cumulative-70% EFC model inputs

The site directories `CLBJ_cumulative70`, `YELL_cumulative70`,
`TALL_cumulative70`, `JERC_cumulative70`, `WOOD_cumulative70`,
`TEAK_cumulative70`, and `WREF_cumulative70` contain:

Additional site directories generated with the same workflow include
`ABBY_cumulative70`, `OSBS_cumulative70`, and `SOAP_cumulative70`.

- one five-layer master feature table;
- eight EFC-specific model-input tables with predictors in trained order;
- a five-row NetCDF-versus-CSV soil-carbon consistency audit; and
- a JSON provenance and coverage audit.

Soil moisture, soil temperature, and historical 7/30/365-day metrics are
calculated separately for the first five CLM soil nodes. When multiple run
subfolders exist, only the path containing `final` is used. Sand and clay are
read from `PCT_SAND` and `PCT_CLAY` in the corresponding surfdata NetCDF.
Soil C (%) and soil N (%) are `carbonTot / 10` and `nitrogenTot / 10` from
the containing biogeochemical horizon in each site surface CSV; C/N is
`carbonTot / nitrogenTot`.

The carbon audit reconstructs surfdata `ORGANIC` as
`bulkDensExclCoarseFrag * carbonTot / 0.58`. Because CSV carbon is reported
to 0.1 g/kg, differences within 0.05 soil-C percentage points are classified
as rounding-consistent. A failed audit does not replace the requested CSV
chemistry; it flags that NetCDF `ORGANIC` was not generated from the same
CSV carbon/bulk-density combination.

If a CSV horizon reports missing or nonpositive `nitrogenTot`, nitrogen is
linearly interpolated at the CLM node depth from positive `nitrogenTot`
measurements at neighboring horizon midpoints. The layer audit records every
interpolated value. This rule was required for OSBS layers 4 and 5.

Regenerate with:

```bash
/opt/anaconda3/envs/neon-efc/bin/python build_multisite_layer_model_inputs.py
```
