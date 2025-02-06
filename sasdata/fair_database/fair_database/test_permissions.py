import os

from django.contrib.auth.models import User
from rest_framework import status
from rest_framework.test import APIClient, APITestCase

from data.models import Data

def find(filename):
    return os.path.join(os.path.dirname(__file__), "../../example_data/1d_data", filename)

class DataListPermissionsTests(APITestCase):
    ''' Test permissions of data views using user_app for authentication. '''

    def setUp(self):
        self.user = User.objects.create_user(username="testUser", password="secret", id=1,
                                             email="email@domain.com")
        self.user2 = User.objects.create_user(username="testUser2", password="secret", id=2,
                                              email="email2@domain.com")
        unowned_test_data = Data.objects.create(id=1, file_name="cyl_400_40.txt",
                                               is_public=True)
        unowned_test_data.file.save("cyl_400_40.txt", open(find("cyl_400_40.txt"), 'rb'))
        private_test_data = Data.objects.create(id=2, current_user=self.user,
                                                file_name="cyl_400_20.txt", is_public=False)
        private_test_data.file.save("cyl_400_20.txt", open(find("cyl_400_20.txt"), 'rb'))
        public_test_data = Data.objects.create(id=3, current_user=self.user,
                                               file_name="cyl_testdata.txt", is_public=True)
        public_test_data.file.save("cyl_testdata.txt", open(find("cyl_testdata.txt"), 'rb'))
        self.login_data_1 = {
            'username': 'testUser',
            'password': 'secret',
            'email': 'email@domain.com'
        }
        self.login_data_2 = {
            'username': 'testUser2',
            'password': 'secret',
            'email': 'email2@domain.com'
        }

    # Authenticated user can view list of data
    # TODO: change to reflect inclusion of owned private data
    def test_list_authenticated(self):
        self.client.post('/auth/login/', data=self.login_data_1)
        response = self.client.get('/v1/data/list/')
        response2 = self.client.get('/v1/data/list/testUser/')
        self.assertEqual(response.data,
                         {"public_data_ids": {1: "cyl_400_40.txt", 3: "cyl_testdata.txt"}})
        self.assertEqual(response2.data,
                         {"user_data_ids": {2: "cyl_400_20.txt", 3: "cyl_testdata.txt"}})

    # Authenticated user cannot view other users' private data on list
    # TODO: Change response codes
    def test_list_authenticated_2(self):
        self.client.post('/auth/login/', data=self.login_data_2)
        response = self.client.get('/v1/data/list/')
        response2 = self.client.get('/v1/data/list/testUser/')
        response3 = self.client.get('/v1/data/list/testUser2/')
        self.assertEqual(response.data,
                         {"public_data_ids": {1: "cyl_400_40.txt", 3: "cyl_testdata.txt"}})
        self.assertEqual(response2.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response3.data, {"user_data_ids": {}})

    # Unauthenticated user can view list of public data
    def test_list_unauthenticated(self):
        response = self.client.get('/v1/data/list/')
        response2 = self.client.get('/v1/data/list/testUser/')
        self.assertEqual(response.data,
                         {"public_data_ids": {1: "cyl_400_40.txt", 3: "cyl_testdata.txt"}})
        self.assertEqual(response2.status_code, status.HTTP_400_BAD_REQUEST)

    # Authenticated user can load public data and owned private data
    def test_load_authenticated(self):
        self.client.post('/auth/login/', data=self.login_data_1)
        response = self.client.get('/v1/data/load/1/')
        response2 = self.client.get('/v1/data/load/2/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response2.status_code, status.HTTP_200_OK)

    # Authenticated user cannot load others' private data
    def test_load_unauthorized(self):
        self.client.post('/auth/login/', data=self.login_data_2)
        response = self.client.get('/v1/data/load/2/')
        response2 = self.client.get('/v1/data/load/3/')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response2.status_code, status.HTTP_200_OK)

    # Unauthenticated user can load public data only
    def test_load_unauthenticated(self):
        response = self.client.get('/v1/data/load/1/')
        response2 = self.client.get('/v1/data/load/2/')
        response3 = self.client.get('/v1/data/load/3/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response2.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response3.status_code, status.HTTP_200_OK)

    # Authenticated user can upload data

    # ***Unauthenticated user can upload public data

    # Unauthenticated user cannot upload private data

    # Authenticated user can update own public data

    # Authenticated user can update own private data

    # Authenticated user cannot update unowned public data

    # Unauthenticated user cannot update data

    # Anyone can download public data

    # Authenticated user can download own data

    # Authenticated user cannot download others' data

    # Unauthenticated user cannot download private data
