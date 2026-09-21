import unittest
from ogura.textdet.prepare_experiment import grouped_splits


class ExperimentTests(unittest.TestCase):
    def test_documents_and_transitive_similarity_never_cross_splits(self):
        pages=[{'source_pdf':f'doc{i}.pdf','page':j} for i in range(20) for j in (1,2)]
        pairs=[('doc0.pdf','excluded.pdf'),('excluded.pdf','doc1.pdf')]
        splits=grouped_splits(pages,pairs)
        self.assertEqual(splits,grouped_splits(pages,pairs))
        locations={}
        for split,rows in splits.items():
            for row in rows:
                locations.setdefault(row['source_pdf'],set()).add(split)
        self.assertTrue(all(len(v)==1 for v in locations.values()))
        self.assertEqual(locations['doc0.pdf'],locations['doc1.pdf'])
        self.assertEqual(sum(map(len,splits.values())),len(pages))

    def test_too_few_independent_groups_rejected(self):
        with self.assertRaises(ValueError):
            grouped_splits([{'source_pdf':'one.pdf','page':1}],[])
