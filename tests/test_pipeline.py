"""Tests for the data pipeline, model variants and the audit utilities.

Dataset-dependent tests skip cleanly when the external download has not been made, so
`python -m unittest discover -s tests` still works on a fresh clone.
"""
import math
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import pandas as pd  # noqa: E402

from pipeline import qualifications  # noqa: E402
from pipeline.models import TfidfVariant  # noqa: E402
from pipeline.skills_normalizer import (  # noqa: E402
    canonical_city, canonical_skill, canonical_skill_list, canonical_state, is_indian_location,
)
from scripts import audit_dataset  # noqa: E402

PROCESSED = 'data/processed/jobs_sample.csv'
CATALOG = 'data/internship.csv'


class SkillNormalisationTests(unittest.TestCase):

    def test_excel_variants(self):
        for raw in ('Excel', 'MS Excel', 'Microsoft Excel', '  EXCEL  '):
            self.assertEqual(canonical_skill(raw), 'excel', raw)

    def test_basic_computer_variants(self):
        for raw in ('Basic Computer', 'Computer Basics', 'basic computer knowledge'):
            self.assertEqual(canonical_skill(raw), 'basic computer', raw)

    def test_sql_java_python_variants(self):
        self.assertEqual(canonical_skill('MySQL'), 'sql')
        self.assertEqual(canonical_skill('PostgreSQL'), 'sql')
        self.assertEqual(canonical_skill('Core Java'), 'java')
        self.assertEqual(canonical_skill('Python3'), 'python')

    def test_data_entry_communication_tally(self):
        self.assertEqual(canonical_skill('Data Entry Operator'), 'data entry')
        self.assertEqual(canonical_skill('Communication Skills'), 'communication')
        self.assertEqual(canonical_skill('Tally ERP 9'), 'tally')

    def test_unknown_skill_is_cleaned_not_dropped(self):
        self.assertEqual(canonical_skill('AutoCAD  '), 'autocad')

    def test_list_normalisation_dedupes(self):
        self.assertEqual(canonical_skill_list('Excel, MS Excel, python, PYTHON'),
                         ['excel', 'python'])

    def test_empty_skill_list(self):
        self.assertEqual(canonical_skill_list(''), [])
        self.assertEqual(canonical_skill_list(None), [])


class LocationNormalisationTests(unittest.TestCase):

    def test_city_aliases(self):
        self.assertEqual(canonical_city('Bangalore'), 'Bengaluru')
        self.assertEqual(canonical_city('bengaluru'), 'Bengaluru')
        self.assertEqual(canonical_city('Delhi NCR'), 'Delhi')
        self.assertEqual(canonical_city('New Delhi'), 'Delhi')
        self.assertEqual(canonical_city('Gurgaon'), 'Gurugram')

    def test_state_alias(self):
        self.assertEqual(canonical_state('orissa'), 'Odisha')
        self.assertEqual(canonical_state('Karnataka'), 'Karnataka')

    def test_indian_detection(self):
        self.assertTrue(is_indian_location('Pune', 'Maharashtra', 'India'))
        self.assertTrue(is_indian_location('Bangalore', '', ''))
        self.assertFalse(is_indian_location('Little River', 'SC', 'United States'))


class QualificationTests(unittest.TestCase):

    def test_requested_labels_all_map(self):
        expected = {
            '10th Pass': 0, '12th Pass': 1, 'ITI': 2, 'Diploma': 2,
            'Undergraduate': 3, 'B.Tech': 3, 'Graduate': 3, 'Postgraduate': 4,
        }
        for label, level in expected.items():
            self.assertEqual(qualifications.level_of(label), level, label)

    def test_freeform_text_maps(self):
        self.assertEqual(qualifications.normalize_qualification('Pursuing B.Tech in CS'), 'B.Tech')
        self.assertEqual(qualifications.normalize_qualification('B.E.'), 'B.Tech')
        self.assertEqual(qualifications.normalize_qualification('ITI Welder'), 'ITI')
        self.assertEqual(qualifications.normalize_qualification('Diploma in Electronics'), 'Diploma')
        self.assertEqual(qualifications.normalize_qualification('MCA / MSc'), 'Postgraduate')

    def test_unrecognised_is_unknown_not_defaulted(self):
        self.assertIsNone(qualifications.normalize_qualification('B.Voc in Retail'))
        self.assertIsNone(qualifications.normalize_qualification(''))
        self.assertIsNone(qualifications.level_of(None))

    def test_hierarchy_rules(self):
        self.assertTrue(qualifications.is_eligible('B.Tech', 'Diploma'))
        self.assertTrue(qualifications.is_eligible('Postgraduate', 'B.Tech'))
        self.assertFalse(qualifications.is_eligible('10th Pass', '12th Pass'))
        self.assertFalse(qualifications.is_eligible('12th Pass', 'Diploma'))

    def test_tracks_are_not_interchangeable(self):
        # ITI must not satisfy a Diploma requirement ...
        self.assertFalse(qualifications.is_eligible('ITI', 'Diploma'))
        # ... but a Diploma holder can take an ITI-level role.
        self.assertTrue(qualifications.is_eligible('Diploma', 'ITI'))
        # A B.Tech holder outranks the vocational track and is eligible either way.
        self.assertTrue(qualifications.is_eligible('B.Tech', 'ITI'))

    def test_unknown_either_side_returns_none(self):
        self.assertIsNone(qualifications.is_eligible('B.Tech', None))
        self.assertIsNone(qualifications.is_eligible(None, 'B.Tech'))


class AuditUtilityTests(unittest.TestCase):

    def test_near_duplicate_detection(self):
        base = 'we are hiring a python developer to build data pipelines in bangalore'
        same = base + ' with sql'
        different = 'we need a retail assistant for a store in pune handling inventory'
        duplicates, evaluated, unique = audit_dataset.near_duplicate_rate(
            [base, same, different], threshold=0.5)
        self.assertEqual(evaluated, 3)
        self.assertEqual(duplicates, 1)
        self.assertEqual(unique, 2)

    def test_html_stripping_and_normalisation(self):
        self.assertEqual(audit_dataset.normalise('<p>Hello   World</p>'), 'hello world')


class TfidfModelTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.frame = pd.read_csv(CATALOG, dtype=str).fillna('')
        cls.model = TfidfVariant('test', ('title', 'required_skills')).fit(cls.frame)

    def test_scores_shape_and_range(self):
        scores = self.model.scores('Python, SQL')
        self.assertEqual(len(scores), len(self.frame))
        self.assertGreaterEqual(float(scores.min()), 0.0)
        self.assertLessEqual(float(scores.max()), 1.0 + 1e-9)

    def test_unknown_query_scores_zero(self):
        self.assertEqual(float(self.model.scores('zzz unknown token').max()), 0.0)
        self.assertEqual(float(self.model.scores('').max()), 0.0)

    def test_exact_match_ranks_first(self):
        scores = self.model.scores('Python, Java, Git, SQL, Data Structures')
        best = int(scores.argmax())
        self.assertEqual(self.frame.iloc[best]['internship_id'], 'INT-IT-25-001')

    def test_inference_is_deterministic(self):
        first = self.model.scores('Excel, Data Entry')
        second = self.model.scores('Excel, Data Entry')
        self.assertTrue((first == second).all())

    def test_missing_columns_are_tolerated(self):
        frame = self.frame[['internship_id', 'title']].head(5).copy()
        model = TfidfVariant('minimal', ('title', 'required_skills')).fit(frame)
        self.assertEqual(len(model.scores('intern')), 5)


class EvaluationMetricTests(unittest.TestCase):

    def test_ndcg_perfect_and_worst(self):
        from scripts.experiment_models import ndcg_at_k
        relevant = {'A', 'B', 'C'}
        self.assertAlmostEqual(ndcg_at_k(['A', 'B', 'C'], relevant), 1.0)
        self.assertAlmostEqual(ndcg_at_k(['X', 'Y', 'Z'], relevant), 0.0)
        value = ndcg_at_k(['X', 'A'], relevant)
        self.assertTrue(0 < value < 1)
        ideal = 1 + 1 / math.log2(3) + 1 / math.log2(4)   # three relevant labels, k=5
        self.assertAlmostEqual(value, (1 / math.log2(3)) / ideal)


@unittest.skipUnless(os.path.exists(PROCESSED), 'processed sample not downloaded')
class ProcessedDatasetTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.frame = pd.read_csv(PROCESSED, dtype=str).fillna('')

    def test_schema(self):
        required = {'job_id', 'title', 'company_name', 'required_skills', 'required_education',
                    'education_known', 'sector', 'location', 'job_type', 'source',
                    'source_url', 'data_status', 'raw_required_education', 'raw_skills'}
        self.assertTrue(required <= set(self.frame.columns), required - set(self.frame.columns))

    def test_no_missing_job_ids_and_unique(self):
        self.assertTrue((self.frame['job_id'] != '').all())
        self.assertTrue(self.frame['job_id'].is_unique)

    def test_education_known_flag_matches_value(self):
        known = self.frame['education_known'].astype(str).str.lower() == 'true'
        self.assertTrue((self.frame.loc[known, 'required_education'] != '').all())
        self.assertTrue((self.frame.loc[~known, 'required_education'] == '').all())

    def test_records_are_flagged_not_current(self):
        self.assertTrue((self.frame['data_status'] == 'synthetic_templated').all())
        self.assertTrue((self.frame['is_currently_open'].astype(str).str.lower() == 'false').all())

    def test_raw_values_preserved(self):
        self.assertTrue((self.frame['raw_required_education'].str.strip() != '').all())
        self.assertTrue((self.frame['raw_skills'].str.strip() != '').all())

    def test_locations_are_indian(self):
        self.assertTrue(is_indian_location('Pune', 'Maharashtra', 'India'))
        self.assertTrue((self.frame['location'] != '').all())


class CatalogIsolationTests(unittest.TestCase):

    def test_production_catalog_is_unchanged(self):
        frame = pd.read_csv(CATALOG, dtype=str).fillna('')
        self.assertEqual(len(frame), 115)
        self.assertTrue(frame['internship_id'].is_unique)

    def test_pipeline_does_not_import_flask(self):
        import pipeline.models  # noqa: F401
        import pipeline.qualifications  # noqa: F401
        import pipeline.skills_normalizer  # noqa: F401
        self.assertNotIn('flask', sys.modules.get('pipeline.models').__dict__)


if __name__ == '__main__':
    unittest.main()
