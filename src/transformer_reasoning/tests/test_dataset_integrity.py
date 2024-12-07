import unittest
from datasets import load_from_disk
from transformer_reasoning.utils import get_project_root

class TestDatasetIntegrity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.profiles = load_from_disk(str(get_project_root() / "generated_data/profiles_dataset"))
        cls.bios = load_from_disk(str(get_project_root() / "generated_data/bios/bios_dataset"))
        cls.qa = load_from_disk(str(get_project_root() / "generated_data/qa_dataset"))

    def test_no_brackets_in_bios(self):
        for bio in self.bios['bio']:
            self.assertNotIn('{', bio)
            self.assertNotIn('}', bio)

    def test_best_friend_symmetry(self):
        for profile in self.profiles:
            if profile['best_friend']['name']:
                friend_profile = self.profiles[profile['best_friend']['index']]
                self.assertEqual(profile['name'], friend_profile['best_friend']['name'])

    def test_worst_enemy_symmetry(self):
        for profile in self.profiles:
            if profile['worst_enemy']['name']:
                enemy_profile = self.profiles[profile['worst_enemy']['index']]
                self.assertEqual(profile['name'], enemy_profile['worst_enemy']['name'])

    def test_parent_child_symmetry(self):
        for profile in self.profiles:
            if profile['child']['name']:
                child_profile = self.profiles[profile['child']['index']]
                self.assertEqual(profile['name'], child_profile['parent']['name'])
            if profile['parent']['name']:
                parent_profile = self.profiles[profile['parent']['index']]
                self.assertEqual(profile['name'], parent_profile['child']['name'])

    def test_qa_best_friend_symmetry(self):
        for item in self.qa['train']:
            name = item['questions.question'].split("'s")[0]
            if f"{name}'s best friend's best friend?" in item['questions.question']:
                self.assertIn(item['questions.answer'], item['questions.question'].split("'s")[0], msg=item['questions.question'])

    def test_qa_worst_enemy_symmetry(self):
        for item in self.qa['train']:
            name = item['questions.question'].split("'s")[0]
            if f"{name}'s worst enemy's worst enemy?" in item['questions.question']:
                self.assertIn(item['questions.answer'], item['questions.question'].split("'s")[0])

    def test_qa_parent_child_symmetry(self):
        for item in self.qa['train']:
            name = item['questions.question'].split("'s")[0]
            if f"{name}'s parent's child?" in item['questions.question']:
                self.assertIn(item['questions.answer'], item['questions.question'].split("'s")[0])
            elif f"{name}'s child's parent?" in item['questions.question']:
                self.assertIn(item['questions.answer'], item['questions.question'].split("'s")[0])

if __name__ == '__main__':
    unittest.main()
