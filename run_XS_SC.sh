rm $(ls -I "*.sh" -I "*.py" -I "*.INP" -I "*.img" -I "*.pdb")
python3 Dtrek_SC_rename.py
python3 Dtrek_SC_ints.py
python3 Dtrek_SC_mkgif.py
python3 Dtrek_stitch.py
python3 Dtrek_int.py
xds_par XDS.INP
