import os
import shutil

from django.conf import settings
from django.test import TestCase
from django.contrib.auth.models import User
from rest_framework.test import APIClient, APITestCase
from rest_framework import status

from .models import Data

def find(filename):
    return os.path.join(os.path.dirname(__file__), "../../example_data/1d_data", filename)

class TestLists(TestCase):
    def setUp(self):
        public_test_data = Data.objects.create(id = 1, file_name = "cyl_400_40.txt",
            is_public = True)
        public_test_data.file.save("cyl_400_40.txt", open(find("cyl_400_40.txt"), 'rb'))
        self.user = User.objects.create_user(username="testUser", password="secret", id = 2)
        private_test_data = Data.objects.create(id = 3, current_user = self.user,
            file_name = "cyl_400_20.txt", is_public = False)
        private_test_data.file.save("cyl_400_20.txt", open(find("cyl_400_20.txt"), 'rb'))
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)

    # Test list public data
    def test_does_list_public(self):
        request = self.client.get('/v1/data/list/')
        self.assertEqual(request.data, {"public_data_ids":{1:"cyl_400_40.txt"}})

    # Test list a user's private data
    def test_does_list_user(self):
        request = self.client.get('/v1/data/list/testUser/', user = self.user)
        self.assertEqual(request.data, {"user_data_ids":{3:"cyl_400_20.txt"}})

    # Test loading a public data file
    def test_does_load_data_info_public(self):
        request = self.client.get('/v1/data/load/1/')
        self.assertEqual(request.status_code, status.HTTP_200_OK)

    # Test loading private data with authorization
    def test_does_load_data_info_private(self):
        request = self.client.get('/v1/data/load/3/')
        self.assertEqual(request.status_code, status.HTTP_200_OK)

    def tearDown(self):
        shutil.rmtree(settings.MEDIA_ROOT)

class TestingDatabase(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="testUser", password="secret", id = 1)
        self.data = Data.objects.create(id = 2, current_user = self.user,
            file_name = "cyl_400_20.txt", is_public = False)
        self.data.file.save("cyl_400_20.txt", open(find("cyl_400_20.txt"), 'rb'))
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)
        self.client2 = APIClient()

    # Test data upload creates data in database
    def test_is_data_being_created(self):
        file = open(find("cyl_400_40.txt"), 'rb')
        data = {
            "is_public":False,
            "file":file
        }
        request = self.client.post('/v1/data/upload/', data=data)
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(request.data, {"current_user":'testUser', "authenticated" : True,
            "file_id" : 3, "file_alternative_name":"cyl_400_40.txt","is_public" : False})
        Data.objects.get(id = 3).delete()

    # Test data upload w/out authenticated user
    def test_is_data_being_created_no_user(self):
        file = open(find("cyl_400_40.txt"), 'rb')
        data = {
            "is_public":False,
            "file":file
        }
        request = self.client2.post('/v1/data/upload/', data=data)
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(request.data, {"current_user":'', "authenticated" : False,
            "file_id" : 3, "file_alternative_name":"cyl_400_40.txt","is_public" : False})
        Data.objects.get(id = 3).delete()

    # Test updating file
    def test_does_file_upload_update(self):
        file = open(find("cyl_400_40.txt"))
        data = {
            "file":file,
            "is_public":False
        }
        request = self.client.put('/v1/data/upload/2/', data = data)
        self.assertEqual(request.data, {"current_user":'testUser', "authenticated" : True,
            "file_id" : 2, "file_alternative_name":"cyl_400_40.txt","is_public" : False})
        Data.objects.get(id = 2).delete()

    # Test file upload update fails when unauthorized
    def test_unauthorized_file_upload_update(self):
        file = open(find("cyl_400_40.txt"))
        data = {
            "file": file,
            "is_public": False
        }
        request = self.client2.put('/v1/data/upload/2/', data=data)
        self.assertEqual(request.status_code, status.HTTP_403_FORBIDDEN)
        Data.objects.get(id=2).delete()

    # Test file download
    def test_does_download(self):
        request = self.client.get('/v1/data/2/download/')
        file_contents = b''.join(request.streaming_content)
        test_file = open(find('cyl_400_20.txt'), 'rb')
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(file_contents, test_file.read())

    # Test file download fails when unauthorized
    def test_unauthorized_download(self):
        request2 = self.client2.get('/v1/data/2/download/')
        self.assertEqual(request2.status_code, status.HTTP_403_FORBIDDEN)

    def tearDown(self):
        shutil.rmtree(settings.MEDIA_ROOT)