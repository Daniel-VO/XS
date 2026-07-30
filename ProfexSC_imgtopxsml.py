"""
Created 30. Juli 2026 by Daniel Van Opdenbosch,
Technical University of Munich

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import fabio
import numpy as np

def take(img, headerkey, indices):
    return np.fromstring(
        img.header[headerkey],
        sep=" "
    )[indices]

def is_transmission(img):
    return (
        "Transmission"
        in img.header.get(
            "CRYSTAL_GONIO_ATT_DESCRIPTION",
            "",
        )
    )

def make_pxsml_string(img):

    detdist = float(
        take(img, "PXD_GONIO_VALUES", -1)
    )

    beamcenter_x, beamcenter_y, pxsize_x, pxsize_y = take(
        img,
        "PXD_SPATIAL_DISTORTION_INFO",
        [0, 1, 2, 3],
    )

    wavelength_angstrom = float(
        take(img, "SOURCE_WAVELENGTH", -1)
    )

    wavelength_nm = wavelength_angstrom / 10

    sample_size = 1.5

    root = ET.Element(
        "ProfexSynchrotronConfiguration",
        model="Chernyshov",
        version="5.5",
    )

    ET.SubElement(
        root,
        "Range",
        unit="degrees",
        start="1",
        end="60",
        step="5",
        shape="0.5",
    )

    beamline = ET.SubElement(
        root,
        "BeamlineParameters",
    )

    ET.SubElement(
        beamline,
        "Sample",
        geometry="CAPILLARY",
    )

    node = ET.SubElement(
        beamline,
        "DetectorDistance",
        unit="mm",
        fit="true",
    )
    node.text = f"{detdist:.3f}"

    node = ET.SubElement(
        beamline,
        "PixelSize",
        unit="mm",
        fit="true",
    )
    node.text = f"{pxsize_x:.4f}"

    node = ET.SubElement(
        beamline,
        "SampleSize",
        unit="mm",
        fit="true",
    )
    node.text = f"{sample_size:.4f}"

    node = ET.SubElement(
        beamline,
        "SensitiveLayer",
        unit="mm",
        fit="true",
    )
    node.text = "1"

    node = ET.SubElement(
        beamline,
        "BeamDivergence",
        unit="degrees",
        fit="true",
        focused="false",
    )
    node.text = "0.03"

    node = ET.SubElement(
        beamline,
        "DetectorTilt",
        unit="degrees",
    )
    node.text = "0"

    corr = ET.SubElement(
        beamline,
        "PositionalCorrections",
        mode="0",
        cutoff="true",
    )

    node = ET.SubElement(
        corr,
        "Wavelength",
        unit="nm",
    )
    node.text = f"{wavelength_nm:.8f}"

    detector = ET.SubElement(
        corr,
        "Detector",
    )

    ET.SubElement(
        detector,
        "DetectorMaterial",
    ).text = "Cd Te"

    ET.SubElement(
        detector,
        "DetectorDensity",
        unit="g/cm3",
    ).text = "5.86"

    ET.SubElement(
        detector,
        "DetectorAbsorptionCoefficient",
        unit="cm-1",
    ).text = "1422.3"

    ET.SubElement(root, "SupportPeaks")
    ET.SubElement(root, "Profiles")

    ET.indent(
        ET.ElementTree(root),
        space="    ",
    )

    return ET.tostring(
        root,
        encoding="unicode",
        xml_declaration=False,
    )

img_files = sorted(
    Path(".").glob("*.img")
)

generated = []

for imgfile in img_files:

    img = fabio.open(str(imgfile))

    if not is_transmission(img):
        continue

    generated.append(
        (
            imgfile,
            make_pxsml_string(img),
        )
    )

if generated:

    first = generated[0][1]

    if any(
        pxsml != first
        for _, pxsml in generated[1:]
    ):

        answer = input(
            "Generated PXSML files differ.\n"
            "Create individual PXSML files instead? [y/N] "
        )

        if answer.lower().startswith("y"):

            for imgfile, pxsml in generated:

                out = imgfile.with_name(
                    f"ProfexSC_{imgfile.stem}"
                ).with_suffix(".pxsml")

                xml = (
                    '<?xml version="1.0" '
                    'encoding="UTF-8"?>\n'
                    + pxsml
                )

                out.write_text(
                    xml,
                    encoding="utf-8",
                )

                print(
                    f"created {out.name}"
                )

            raise SystemExit

        raise RuntimeError(
            "generated PXSML files are not identical"
        )

    folder = Path(".").resolve().name

    out = Path(
        f"ProfexSC_{folder}.pxsml"
    )

    xml = (
        '<?xml version="1.0" '
        'encoding="UTF-8"?>\n'
        + first
    )

    out.write_text(
        xml,
        encoding="utf-8",
    )

    print(
        f"created {out.name}"
    )
