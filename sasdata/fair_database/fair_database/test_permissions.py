import os
import shutil

from django.conf import settings
from django.contrib.auth.models import User
from rest_framework import status
from rest_framework.test import APITestCase

from data.models import DataFile


def find(filename):
    return os.path.join(
        os.path.dirname(__file__), "../../example_data/1d_data", filename
    )


def auth_header(response):
    return {"Authorization": "Token " + response.data["token"]}


class DataListPermissionsTests(APITestCase):
    """Test permissions of data views using user_app for authentication."""

    def setUp(self):
        self.user = User.objects.create_user(
            username="testUser", password="secret", id=1, email="email@domain.com"
        )
        self.user2 = User.objects.create_user(
            username="testUser2", password="secret", id=2, email="email2@domain.com"
        )
        unowned_test_data = DataFile.objects.create(
            id=1, file_name="cyl_400_40.txt", is_public=True
        )
        unowned_test_data.file.save(
            "cyl_400_40.txt", open(find("cyl_400_40.txt"), "rb")
        )
        private_test_data = DataFile.objects.create(
            id=2, current_user=self.user, file_name="cyl_400_20.txt", is_public=False
        )
        private_test_data.file.save(
            "cyl_400_20.txt", open(find("cyl_400_20.txt"), "rb")
        )
        public_test_data = DataFile.objects.create(
            id=3, current_user=self.user, file_name="cyl_testdata.txt", is_public=True
        )
        public_test_data.file.save(
            "cyl_testdata.txt", open(find("cyl_testdata.txt"), "rb")
        )
        self.login_data_1 = {
            "username": "testUser",
            "password": "secret",
            "email": "email@domain.com",
        }
        self.login_data_2 = {
            "username": "testUser2",
            "password": "secret",
            "email": "email2@domain.com",
        }

    # Authenticated user can view list of data
    # TODO: change to reflect inclusion of owned private data
    def test_list_authenticated(self):
        token = self.client.post("/auth/login/", data=self.login_data_1)
        response = self.client.get("/v1/data/list/", headers=auth_header(token))
        response2 = self.client.get(
            "/v1/data/list/testUser/", headers=auth_header(token)
        )
        self.assertEqual(
            response.data,
            {"public_data_ids": {1: "cyl_400_40.txt", 3: "cyl_testdata.txt"}},
        )
        self.assertEqual(
            response2.data,
            {"user_data_ids": {2: "cyl_400_20.txt", 3: "cyl_testdata.txt"}},
        )

    # Authenticated user cannot view other users' private data on list
    # TODO: Change response codes
    def test_list_authenticated_2(self):
        token = self.client.post("/auth/login/", data=self.login_data_2)
        response = self.client.get("/v1/data/list/", headers=auth_header(token))
        response2 = self.client.get(
            "/v1/data/list/testUser/", headers=auth_header(token)
        )
        response3 = self.client.get(
            "/v1/data/list/testUser2/", headers=auth_header(token)
        )
        self.assertEqual(
            response.data,
            {"public_data_ids": {1: "cyl_400_40.txt", 3: "cyl_testdata.txt"}},
        )
        self.assertEqual(response2.status_code, status.HTTP_200_OK)
        self.assertEqual(response2.data, {"user_data_ids": {3: "cyl_testdata.txt"}})
        self.assertEqual(response3.data, {"user_data_ids": {}})

    # Unauthenticated user can view list of public data
    def test_list_unauthenticated(self):
        response = self.client.get("/v1/data/list/")
        response2 = self.client.get("/v1/data/list/testUser/")
        self.assertEqual(
            response.data,
            {"public_data_ids": {1: "cyl_400_40.txt", 3: "cyl_testdata.txt"}},
        )
        self.assertEqual(response2.status_code, status.HTTP_200_OK)
        self.assertEqual(response2.data, {"user_data_ids": {3: "cyl_testdata.txt"}})

    # Authenticated user can load public data and owned private data
    def test_load_authenticated(self):
        token = self.client.post("/auth/login/", data=self.login_data_1)
        response = self.client.get("/v1/data/load/1/", headers=auth_header(token))
        response2 = self.client.get("/v1/data/load/2/", headers=auth_header(token))
        response3 = self.client.get("/v1/data/load/3/", headers=auth_header(token))
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response2.status_code, status.HTTP_200_OK)
        self.assertEqual(response3.status_code, status.HTTP_200_OK)

    # Authenticated user cannot load others' private data
    def test_load_unauthorized(self):
        token = self.client.post("/auth/login/", data=self.login_data_2)
        response = self.client.get("/v1/data/load/2/", headers=auth_header(token))
        response2 = self.client.get("/v1/data/load/3/", headers=auth_header(token))
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(response2.status_code, status.HTTP_200_OK)

    # Unauthenticated user can load public data only
    def test_load_unauthenticated(self):
        response = self.client.get("/v1/data/load/1/")
        response2 = self.client.get("/v1/data/load/2/")
        response3 = self.client.get("/v1/data/load/3/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response2.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(response3.status_code, status.HTTP_200_OK)

    # Authenticated user can upload data
    def test_upload_authenticated(self):
        token = self.client.post("/auth/login/", data=self.login_data_1)
        file = open(find("cyl_testdata1.txt"), "rb")
        data = {"file": file, "is_public": False}
        response = self.client.post(
            "/v1/data/upload/", data=data, headers=auth_header(token)
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data,
            {
                "current_user": "testUser",
                "authenticated": True,
                "file_id": 4,
                "file_alternative_name": "cyl_testdata1.txt",
                "is_public": False,
            },
        )
        DataFile.objects.get(id=4).delete()

    # Unauthenticated user can upload public data only
    def test_upload_unauthenticated(self):
        file = open(find("cyl_testdata2.txt"), "rb")
        file2 = open(find("cyl_testdata2.txt"), "rb")
        data = {"file": file, "is_public": True}
        data2 = {"file": file2, "is_public": False}
        response = self.client.post("/v1/data/upload/", data=data)
        response2 = self.client.post("/v1/data/upload/", data=data2)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data,
            {
                "current_user": "",
                "authenticated": False,
                "file_id": 4,
                "file_alternative_name": "cyl_testdata2.txt",
                "is_public": True,
            },
        )
        self.assertEqual(response2.status_code, status.HTTP_400_BAD_REQUEST)

    # Authenticated user can update own data
    def test_upload_put_authenticated(self):
        token = self.client.post("/auth/login/", data=self.login_data_1)
        data = {"is_public": False}
        response = self.client.put(
            "/v1/data/upload/2/", data=data, headers=auth_header(token)
        )
        response2 = self.client.put(
            "/v1/data/upload/3/", data=data, headers=auth_header(token)
        )
        self.assertEqual(
            response.data,
            {
                "current_user": "testUser",
                "authenticated": True,
                "file_id": 2,
                "file_alternative_name": "cyl_400_20.txt",
                "is_public": False,
            },
        )
        self.assertEqual(
            response2.data,
            {
                "current_user": "testUser",
                "authenticated": True,
                "file_id": 3,
                "file_alternative_name": "cyl_testdata.txt",
                "is_public": False,
            },
        )
        DataFile.objects.get(id=3).is_public = True

    # Authenticated user cannot update unowned data
    def test_upload_put_unauthorized(self):
        token = self.client.post("/auth/login/", data=self.login_data_2)
        file = open(find("cyl_400_40.txt"))
        data = {"file": file, "is_public": False}
        response = self.client.put(
            "/v1/data/upload/1/", data=data, headers=auth_header(token)
        )
        response2 = self.client.put(
            "/v1/data/upload/2/", data=data, headers=auth_header(token)
        )
        response3 = self.client.put(
            "/v1/data/upload/3/", data=data, headers=auth_header(token)
        )
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(response2.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(response3.status_code, status.HTTP_403_FORBIDDEN)

    # Unauthenticated user cannot update data
    def test_upload_put_unauthenticated(self):
        file = open(find("cyl_400_40.txt"))
        data = {"file": file, "is_public": False}
        response = self.client.put("/v1/data/upload/1/", data=data)
        response2 = self.client.put("/v1/data/upload/2/", data=data)
        response3 = self.client.put("/v1/data/upload/3/", data=data)
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(response2.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(response3.status_code, status.HTTP_403_FORBIDDEN)

    # Authenticated user can download public and own data
    def test_download_authenticated(self):
        token = self.client.post("/auth/login/", data=self.login_data_1)
        response = self.client.get("/v1/data/1/download/", headers=auth_header(token))
        response2 = self.client.get("/v1/data/2/download/", headers=auth_header(token))
        response3 = self.client.get("/v1/data/3/download/", headers=auth_header(token))
        b"".join(response.streaming_content)
        b"".join(response2.streaming_content)
        b"".join(response3.streaming_content)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response2.status_code, status.HTTP_200_OK)
        self.assertEqual(response3.status_code, status.HTTP_200_OK)

    # Authenticated user cannot download others' data
    def test_download_unauthorized(self):
        token = self.client.post("/auth/login/", data=self.login_data_2)
        response = self.client.get("/v1/data/2/download/", headers=auth_header(token))
        response2 = self.client.get("/v1/data/3/download/", headers=auth_header(token))
        b"".join(response2.streaming_content)
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(response2.status_code, status.HTTP_200_OK)

    # Unauthenticated user cannot download private data
    def test_download_unauthenticated(self):
        response = self.client.get("/v1/data/1/download/")
        response2 = self.client.get("/v1/data/2/download/")
        response3 = self.client.get("/v1/data/3/download/")
        b"".join(response.streaming_content)
        b"".join(response3.streaming_content)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response2.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(response3.status_code, status.HTTP_200_OK)

    def tearDown(self):
        shutil.rmtree(settings.MEDIA_ROOT)
