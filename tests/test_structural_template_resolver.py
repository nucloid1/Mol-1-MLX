# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Ported from the cytokineking/Protenix fork: https://github.com/cytokineking/Protenix
# Fork revision: 5847e387bf7caac4716a5fc0f6da391f9c85cf53
# Fork file: tests/test_structural_template_resolver.py
# Changes: imports point at the mol package; as_protenix_dict is as_mol_dict;
# TemplateHitFeaturizer takes no kalign path; TemplateEmbedder mask tests added.
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mol.data.constants import ATOM37_ORDER, PROTEIN_CHAIN
from mol.data.template.structural_template import (
    StructuralTemplateError,
    features_from_template_spec_for_chain,
    parse_template_file,
    resolve_task_structural_templates,
)
from mol.data.template.template_featurizer import (
    TemplateFeatureAssemblyLine,
    get_safe_entity_id_for_template_copy,
)
from mol.data.template.template_utils import TemplateHitFeaturizer
from mol.model.modules.pairformer import TemplateEmbedder


def _pdb_atom_line(
    serial: int,
    atom_name: str,
    res_name: str,
    chain_id: str,
    res_id: int,
    x: float,
    y: float,
    z: float,
) -> str:
    return (
        f"ATOM  {serial:5d} {atom_name:<4} {res_name:>3} {chain_id:1}"
        f"{res_id:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
        "  1.00 20.00           "
        f"{atom_name[0]:>2}"
    )


def _write_pdb(path: Path, chains: dict[str, list[str]]) -> None:
    aa3 = {
        "A": "ALA",
        "C": "CYS",
        "D": "ASP",
        "E": "GLU",
        "F": "PHE",
        "G": "GLY",
        "H": "HIS",
        "I": "ILE",
        "K": "LYS",
        "L": "LEU",
    }
    lines = []
    serial = 1
    chain_offset = 0.0
    for chain_id, sequence in chains.items():
        for res_idx, residue in enumerate(sequence, start=1):
            res_name = aa3[residue]
            x = chain_offset + res_idx * 3.8
            for atom_name, dx, dy, dz in [
                ("N", 0.0, 0.0, 0.0),
                ("CA", 1.2, 0.1, 0.0),
                ("C", 2.4, 0.0, 0.1),
                ("O", 3.1, 0.0, 0.1),
            ]:
                lines.append(
                    _pdb_atom_line(
                        serial,
                        atom_name,
                        res_name,
                        chain_id,
                        res_idx,
                        x + dx,
                        dy,
                        dz,
                    )
                )
                serial += 1
        lines.append("TER")
        chain_offset += 50.0
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def _write_pdb_with_residue_names(path: Path, chains: dict[str, list[str]]) -> None:
    lines = []
    serial = 1
    chain_offset = 0.0
    for chain_id, residues in chains.items():
        for res_idx, res_name in enumerate(residues, start=1):
            x = chain_offset + res_idx * 3.8
            for atom_name, dx, dy, dz in [
                ("N", 0.0, 0.0, 0.0),
                ("CA", 1.2, 0.1, 0.0),
                ("C", 2.4, 0.0, 0.1),
                ("O", 3.1, 0.0, 0.1),
            ]:
                lines.append(
                    _pdb_atom_line(
                        serial,
                        atom_name,
                        res_name,
                        chain_id,
                        res_idx,
                        x + dx,
                        dy,
                        dz,
                    )
                )
                serial += 1
        lines.append("TER")
        chain_offset += 50.0
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


class TestStructuralTemplateResolver(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)
        self.featurizer = TemplateHitFeaturizer(
            mmcif_dir=str(self.tmp_path),
            template_cache_dir=None,
            _zero_center_positions=True,
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_direct_pdb_template_for_single_chain(self):
        pdb_path = self.tmp_path / "single.pdb"
        _write_pdb(pdb_path, {"H": list("ACDEFG")})

        resolved = features_from_template_spec_for_chain(
            template_spec={"pdb": str(pdb_path)},
            query_sequence="ACDEFG",
            hit_processor=self.featurizer._hit_processor,
            template_index=0,
        )

        feature = resolved.features
        self.assertEqual(feature["template_all_atom_positions"].shape, (6, 37, 3))
        self.assertEqual(feature["template_all_atom_masks"].shape, (6, 37))
        self.assertGreater(feature["template_all_atom_masks"].sum(), 0)
        self.assertEqual(resolved.hit.aligned_cols, 6)

    def test_task_level_multichain_template_with_explicit_mapping(self):
        pdb_path = self.tmp_path / "multi.pdb"
        _write_pdb(pdb_path, {"H": list("ACDEFG"), "L": list("HIKLAC")})

        resolved = resolve_task_structural_templates(
            template_specs=[
                {
                    "pdb": str(pdb_path),
                    "chain_id": ["A", "B"],
                    "template_id": ["H", "L"],
                }
            ],
            target_chains={"A": "ACDEFG", "B": "HIKLAC"},
            hit_processor=self.featurizer._hit_processor,
        )

        self.assertEqual(set(resolved), {"A", "B"})
        self.assertEqual(len(resolved["A"]), 1)
        self.assertEqual(len(resolved["B"]), 1)

    def test_task_level_multichain_template_preserves_shared_frame(self):
        pdb_path = self.tmp_path / "shared_frame.pdb"
        _write_pdb(pdb_path, {"H": list("GGGGGG"), "L": list("GGGGGG")})

        resolved = resolve_task_structural_templates(
            template_specs=[
                {
                    "pdb": str(pdb_path),
                    "chain_id": ["A", "B"],
                    "template_id": ["H", "L"],
                }
            ],
            target_chains={"A": "GGGGGG", "B": "GGGGGG"},
            hit_processor=self.featurizer._hit_processor,
        )

        feature_a = resolved["A"][0]
        feature_b = resolved["B"][0]
        self.assertEqual(
            feature_a["template_complex_id"],
            feature_b["template_complex_id"],
        )
        self.assertEqual(feature_a["template_target_chain_id"], "A")
        self.assertEqual(feature_b["template_target_chain_id"], "B")

        ca_idx = ATOM37_ORDER["CA"]
        distance = np.linalg.norm(
            feature_a["template_all_atom_positions"][0, ca_idx]
            - feature_b["template_all_atom_positions"][0, ca_idx]
        )
        self.assertAlmostEqual(distance, 50.0, places=4)

    def test_grouped_template_assembly_keeps_cross_chain_features(self):
        pdb_path = self.tmp_path / "assembly_complex.pdb"
        _write_pdb(pdb_path, {"H": list("GGGGGG"), "L": list("GGGGGG")})
        resolved = resolve_task_structural_templates(
            template_specs=[
                {
                    "pdb": str(pdb_path),
                    "chain_id": ["A", "B"],
                    "template_id": ["H", "L"],
                }
            ],
            target_chains={"A": "GGGGGG", "B": "GGGGGG"},
            hit_processor=self.featurizer._hit_processor,
        )
        bioassembly = {
            0: {
                "entity_id": 0,
                "chain_id": "A",
                "sequence": "GGGGGG",
                "chain_entity_type": PROTEIN_CHAIN,
                "templates": list(resolved["A"]),
                "copy_templates": False,
            },
            1: {
                "entity_id": 1,
                "chain_id": "B",
                "sequence": "GGGGGG",
                "chain_entity_type": PROTEIN_CHAIN,
                "templates": list(resolved["B"]),
                "copy_templates": False,
            },
        }

        features = (
            TemplateFeatureAssemblyLine(max_templates=4)
            .assemble(bioassembly, np.arange(12))
            .as_mol_dict()
        )

        pair_mask = features["template_pair_geometry_mask"]
        self.assertEqual(pair_mask.shape, (4, 12, 12))
        self.assertEqual(pair_mask[0, 0, 6], 1.0)
        self.assertEqual(pair_mask[0, 6, 0], 1.0)
        self.assertEqual(pair_mask[1, 0, 6], 0.0)
        self.assertEqual(features["template_pseudo_beta_mask"][0, 0, 6], 1.0)
        self.assertEqual(features["template_distogram"][0, 0, 6].sum(), 1.0)

    def test_independent_task_templates_do_not_emit_pair_geometry_mask(self):
        pdb_path = self.tmp_path / "independent.pdb"
        _write_pdb(pdb_path, {"H": list("GGGGGG"), "L": list("GGGGGG")})
        resolved = resolve_task_structural_templates(
            template_specs=[
                {"pdb": str(pdb_path), "chain_id": ["A"], "template_id": ["H"]},
                {"pdb": str(pdb_path), "chain_id": ["B"], "template_id": ["L"]},
            ],
            target_chains={"A": "GGGGGG", "B": "GGGGGG"},
            hit_processor=self.featurizer._hit_processor,
        )
        bioassembly = {
            0: {
                "entity_id": 0,
                "chain_id": "A",
                "sequence": "GGGGGG",
                "chain_entity_type": PROTEIN_CHAIN,
                "templates": list(resolved["A"]),
                "copy_templates": False,
            },
            1: {
                "entity_id": 1,
                "chain_id": "B",
                "sequence": "GGGGGG",
                "chain_entity_type": PROTEIN_CHAIN,
                "templates": list(resolved["B"]),
                "copy_templates": False,
            },
        }

        features = (
            TemplateFeatureAssemblyLine(max_templates=4)
            .assemble(bioassembly, np.arange(12))
            .as_mol_dict()
        )

        self.assertNotIn("template_pair_geometry_mask", features)

    def test_ambiguous_assignment_requires_template_id(self):
        pdb_path = self.tmp_path / "ambiguous.pdb"
        _write_pdb(pdb_path, {"H": list("ACDEFG"), "L": list("ACDEFG")})

        with self.assertRaisesRegex(StructuralTemplateError, "ambiguous"):
            resolve_task_structural_templates(
                template_specs=[{"pdb": str(pdb_path), "chain_id": ["A"]}],
                target_chains={"A": "ACDEFG"},
                hit_processor=self.featurizer._hit_processor,
            )

    def test_duplicate_template_indices_are_rejected(self):
        pdb_path = self.tmp_path / "duplicate_indices.pdb"
        _write_pdb(pdb_path, {"H": list("ACDEFG")})

        with self.assertRaisesRegex(StructuralTemplateError, "Duplicate template index"):
            features_from_template_spec_for_chain(
                template_spec={
                    "pdb": str(pdb_path),
                    "queryIndices": [0, 1],
                    "templateIndices": [0, 0],
                },
                query_sequence="ACDEFG",
                hit_processor=self.featurizer._hit_processor,
                template_index=0,
            )

    def test_pdb_parser_ignores_non_protein_polymer_chains(self):
        pdb_path = self.tmp_path / "protein_and_dna.pdb"
        _write_pdb_with_residue_names(
            pdb_path,
            {"A": ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY"], "B": ["DA", "DT"]},
        )

        mmcif_object = parse_template_file(str(pdb_path))

        self.assertEqual(set(mmcif_object.chain_to_seqres), {"A"})

    def test_task_level_templates_disable_entity_template_copy(self):
        bioassembly = {
            0: {
                "entity_id": 0,
                "sequence": "ACDEFG",
                "copy_templates": False,
            },
            1: {
                "entity_id": 0,
                "sequence": "ACDEFG",
                "copy_templates": True,
            },
        }

        self.assertEqual(get_safe_entity_id_for_template_copy(bioassembly), [])


class TestTemplateEmbedderPairGeometryMask(unittest.TestCase):
    """Inter-chain template geometry reaches the model only through the mask."""

    N_A, N_B = 5, 4

    def setUp(self):
        torch.manual_seed(0)
        self.embedder = TemplateEmbedder(n_blocks=1, c=8, c_z=16).eval()
        n = self.N_A + self.N_B
        self.n = n
        self.z = torch.randn(n, n, 16)
        self.same_chain = torch.zeros(n, n)
        self.same_chain[: self.N_A, : self.N_A] = 1.0
        self.same_chain[self.N_A :, self.N_A :] = 1.0
        self.features = {
            "asym_id": torch.tensor([0] * self.N_A + [1] * self.N_B),
            "template_aatype": torch.randint(0, 20, (1, n)),
            "template_distogram": torch.rand(1, n, n, 39),
            "template_pseudo_beta_mask": torch.ones(1, n, n),
            "template_unit_vector": torch.randn(1, n, n, 3),
            "template_backbone_frame_mask": torch.ones(1, n, n),
        }

    def _embed(self, features):
        with torch.no_grad():
            return self.embedder(dict(features), self.z)

    def _with_new_inter_chain_geometry(self, features):
        changed = dict(features)
        inter = (1.0 - self.same_chain)[None, ..., None]
        changed["template_distogram"] = (
            features["template_distogram"] * (1.0 - inter)
            + torch.rand_like(features["template_distogram"]) * inter
        )
        return changed

    def test_same_chain_mask_matches_the_default(self):
        masked = dict(self.features, template_pair_geometry_mask=self.same_chain[None])
        torch.testing.assert_close(self._embed(masked), self._embed(self.features))

    def test_inter_chain_geometry_is_ignored_without_the_mask(self):
        changed = self._with_new_inter_chain_geometry(self.features)
        torch.testing.assert_close(self._embed(changed), self._embed(self.features))

    def test_inter_chain_geometry_is_used_with_the_mask(self):
        full = torch.ones(1, self.n, self.n)
        base = dict(self.features, template_pair_geometry_mask=full)
        changed = self._with_new_inter_chain_geometry(base)
        diff = (self._embed(changed) - self._embed(base)).abs().max().item()
        self.assertGreater(diff, 1e-4)


if __name__ == "__main__":
    unittest.main()
