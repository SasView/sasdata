import os

from django.test import TestCase
from django.contrib.auth.models import User
from rest_framework.test import APIClient

from .models import Data

def find(filename):
    return os.path.join(os.path.dirname(__file__), "../../example_data/1d_data", filename)

class TestLists(TestCase):
    def setUp(self):
        public_test_data = Data.objects.create(id = 1, file_name = "cyl_400_40.txt", is_public = True)
        public_test_data.file.save("cyl_400_40.txt", open(find("cyl_400_40.txt"), 'rb'))
        self.user = User.objects.create_user(username="testUser", password="secret", id = 2)
        private_test_data = Data.objects.create(id = 3, current_user = self.user, file_name = "cyl_400_20.txt", is_public = False)
        private_test_data.file.save("cyl_400_20.txt", open(find("cyl_400_20.txt"), 'rb'))
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)

    #working
    def test_does_list_public(self):
        request = self.client.get('/v1/data/list/')
        self.assertEqual(request.data, {"public_data_ids":{1:"cyl_400_40.txt"}})

    def test_does_list_user(self):
        request = self.client.get('/v1/data/list/testUser/', user = self.user)
        self.assertEqual(request.data, {"user_data_ids":{3:"cyl_400_20.txt"}})