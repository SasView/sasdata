import os
import shutil

from django.conf import settings
from django.test import TestCase
from django.contrib.auth.models import User
from rest_framework.test import APIClient, APITestCase
from rest_framework import status

from data.models import DataFile


def find(filename):
    return os.path.join(
        os.path.dirname(__file__), "../../example_data/1d_data", filename
    )


class TestLists(TestCase):
    def setUp(self):
        public_test_data = DataFile.objects.create(
            id=1, file_name="cyl_400_40.txt", is_public=True
        )
        public_test_data.file.save("cyl_400_40.txt", open(find("cyl_400_40.txt"), "rb"))
        self.user = User.objects.create_user(
            username="testUser", password="secret", id=2
        )
        private_test_data = DataFile.objects.create(
            id=3, current_user=self.user, file_name="cyl_400_20.txt", is_public=False
        )
        private_test_data.file.save(
            "cyl_400_20.txt", open(find("cyl_400_20.txt"), "rb")
        )
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)

    # Test list public data
    def test_does_list_public(self):
        request = self.client.get("/v1/data/list/")
        self.assertEqual(request.data, {"public_data_ids": {1: "cyl_400_40.txt"}})

    # Test list a user's private data
    def test_does_list_user(self):
        request = self.client.get("/v1/data/list/testUser/", user=self.user)
        self.assertEqual(request.data, {"user_data_ids": {3: "cyl_400_20.txt"}})

    # Test list another user's public data
    def test_list_other_user(self):
        client2 = APIClient()
        request = client2.get("/v1/data/list/testUser/", user=self.user)
        self.assertEqual(request.data, {"user_data_ids": {}})

    # Test list a nonexistent user's data
    def test_list_nonexistent_user(self):
        request = self.client.get("/v1/data/list/fakeUser/")
        self.assertEqual(request.status_code, status.HTTP_404_NOT_FOUND)

    # Test loading a public data file
    def test_does_load_data_info_public(self):
        request = self.client.get("/v1/data/load/1/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)

    # Test loading private data with authorization
    def test_does_load_data_info_private(self):
        request = self.client.get("/v1/data/load/3/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)

    # Test loading data that does not exist
    def test_load_data_info_nonexistent(self):
        request = self.client.get("/v1/data/load/5/")
        self.assertEqual(request.status_code, status.HTTP_404_NOT_FOUND)

    def tearDown(self):
        shutil.rmtree(settings.MEDIA_ROOT)


class TestingDatabase(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="testUser", password="secret", id=1
        )
        self.data = DataFile.objects.create(
            id=2, current_user=self.user, file_name="cyl_400_20.txt", is_public=False
        )
        self.data.file.save("cyl_400_20.txt", open(find("cyl_400_20.txt"), "rb"))
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)
        self.client2 = APIClient()

    # Test data upload creates data in database
    def test_is_data_being_created(self):
        file = open(find("cyl_400_40.txt"), "rb")
        data = {"is_public": False, "file": file}
        request = self.client.post("/v1/data/upload/", data=data)
        self.assertEqual(request.status_code, status.HTTP_201_CREATED)
        self.assertEqual(
            request.data,
            {
                "current_user": "testUser",
                "authenticated": True,
                "file_id": 3,
                "file_alternative_name": "cyl_400_40.txt",
                "is_public": False,
            },
        )
        DataFile.objects.get(id=3).delete()

    # Test data upload w/out authenticated user
    def test_is_data_being_created_no_user(self):
        file = open(find("cyl_400_40.txt"), "rb")
        data = {"is_public": True, "file": file}
        request = self.client2.post("/v1/data/upload/", data=data)
        self.assertEqual(request.status_code, status.HTTP_201_CREATED)
        self.assertEqual(
            request.data,
            {
                "current_user": "",
                "authenticated": False,
                "file_id": 3,
                "file_alternative_name": "cyl_400_40.txt",
                "is_public": True,
            },
        )
        DataFile.objects.get(id=3).delete()

    # Test updating file
    def test_does_file_upload_update(self):
        file = open(find("cyl_400_40.txt"))
        data = {"file": file, "is_public": False}
        request = self.client.put("/v1/data/upload/2/", data=data)
        self.assertEqual(
            request.data,
            {
                "current_user": "testUser",
                "authenticated": True,
                "file_id": 2,
                "file_alternative_name": "cyl_400_40.txt",
                "is_public": False,
            },
        )
        DataFile.objects.get(id=2).delete()

    # Test updating a public file
    def test_public_file_upload_update(self):
        data_object = DataFile.objects.create(
            id=3, current_user=self.user, file_name="cyl_testdata.txt", is_public=True
        )
        data_object.file.save("cyl_testdata.txt", open(find("cyl_testdata.txt"), "rb"))
        file = open(find("cyl_testdata1.txt"))
        data = {"file": file, "is_public": True}
        request = self.client.put("/v1/data/upload/3/", data=data)
        self.assertEqual(
            request.data,
            {
                "current_user": "testUser",
                "authenticated": True,
                "file_id": 3,
                "file_alternative_name": "cyl_testdata1.txt",
                "is_public": True,
            },
        )
        DataFile.objects.get(id=3).delete()

    # Test file upload update fails when unauthorized
    def test_unauthorized_file_upload_update(self):
        file = open(find("cyl_400_40.txt"))
        data = {"file": file, "is_public": False}
        request = self.client2.put("/v1/data/upload/2/", data=data)
        self.assertEqual(request.status_code, status.HTTP_403_FORBIDDEN)
        DataFile.objects.get(id=2).delete()

    # Test update nonexistent file fails
    def test_file_upload_update_not_found(self):
        file = open(find("cyl_400_40.txt"))
        data = {"file": file, "is_public": False}
        request = self.client2.put("/v1/data/upload/5/", data=data)
        self.assertEqual(request.status_code, status.HTTP_404_NOT_FOUND)

    # Test file download
    def test_does_download(self):
        request = self.client.get("/v1/data/2/download/")
        file_contents = b"".join(request.streaming_content)
        test_file = open(find("cyl_400_20.txt"), "rb")
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(file_contents, test_file.read())

    # Test file download fails when unauthorized
    def test_unauthorized_download(self):
        request2 = self.client2.get("/v1/data/2/download/")
        self.assertEqual(request2.status_code, status.HTTP_403_FORBIDDEN)

    # Test download nonexistent file
    def test_download_nonexistent(self):
        request = self.client.get("/v1/data/5/download/")
        self.assertEqual(request.status_code, status.HTTP_404_NOT_FOUND)

    def tearDown(self):
        shutil.rmtree(settings.MEDIA_ROOT)


class TestAccessManagement(TestCase):
    def setUp(self):
        self.user1 = User.objects.create_user(username="testUser", password="secret")
        self.user2 = User.objects.create_user(username="testUser2", password="secret2")
        self.private_test_data = DataFile.objects.create(
            id=1, current_user=self.user1, file_name="cyl_400_40.txt", is_public=False
        )
        self.private_test_data.file.save(
            "cyl_400_40.txt", open(find("cyl_400_40.txt"), "rb")
        )
        self.shared_test_data = DataFile.objects.create(
            id=2, current_user=self.user1, file_name="cyl_400_20.txt", is_public=False
        )
        self.shared_test_data.file.save(
            "cyl_400_20.txt", open(find("cyl_400_20.txt"), "rb")
        )
        self.shared_test_data.users.add(self.user2)
        self.client1 = APIClient()
        self.client1.force_authenticate(self.user1)
        self.client2 = APIClient()
        self.client2.force_authenticate(self.user2)

    # test viewing no one with access
    def test_view_no_access(self):
        request = self.client1.get("/v1/data/manage/1/")
        data = {"file": 1, "file_name": "cyl_400_40.txt", "users": []}
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(request.data, data)

    # test viewing list of users with access
    def test_view_access(self):
        request = self.client1.get("/v1/data/manage/2/")
        data = {"file": 2, "file_name": "cyl_400_20.txt", "users": ["testUser2"]}
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(request.data, data)

    # test granting another user access to private data
    def test_grant_access(self):
        data = {"username": "testUser2", "access": True}
        request1 = self.client1.put("/v1/data/manage/1/", data=data)
        request2 = self.client2.get("/v1/data/load/1/")
        self.assertEqual(request1.status_code, status.HTTP_200_OK)
        self.assertEqual(request2.status_code, status.HTTP_200_OK)

    # test removing another user's access to private data
    def test_remove_access(self):
        data = {"username": "testUser2", "access": False}
        request1 = self.client2.get("/v1/data/load/2/")
        request2 = self.client1.put("/v1/data/manage/2/", data=data)
        request3 = self.client2.get("/v1/data/load/2/")
        self.assertEqual(request1.status_code, status.HTTP_200_OK)
        self.assertEqual(request2.status_code, status.HTTP_200_OK)
        self.assertEqual(request3.status_code, status.HTTP_403_FORBIDDEN)

    def test_remove_no_access(self):
        data = {"username": "testUser2", "access": False}
        request1 = self.client2.get("/v1/data/load/1/")
        request2 = self.client1.put("/v1/data/manage/1/", data=data)
        request3 = self.client2.get("/v1/data/load/1/")
        self.assertEqual(request1.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(request2.status_code, status.HTTP_200_OK)
        self.assertEqual(request3.status_code, status.HTTP_403_FORBIDDEN)

    def test_cant_revoke_own_access(self):
        data = {"username": "testUser", "access": False}
        request1 = self.client1.put("/v1/data/manage/1/", data=data)
        request2 = self.client1.get("/v1/data/load/1/")
        self.assertEqual(request1.status_code, status.HTTP_200_OK)
        self.assertEqual(request2.status_code, status.HTTP_200_OK)

    def test_grant_existing_access(self):
        data = {"username": "testUser2", "access": True}
        request1 = self.client2.get("/v1/data/load/2/")
        request2 = self.client1.put("/v1/data/manage/2/", data=data)
        request3 = self.client2.get("/v1/data/load/2/")
        self.assertEqual(request1.status_code, status.HTTP_200_OK)
        self.assertEqual(request2.status_code, status.HTTP_200_OK)
        self.assertEqual(request3.status_code, status.HTTP_200_OK)

    def test_no_edit_access(self):
        data = {"is_public": True}
        request = self.client2.put("/v1/data/upload/2/", data=data)
        self.assertEqual(request.status_code, status.HTTP_403_FORBIDDEN)
        self.assertFalse(self.shared_test_data.is_public)

    def test_only_view_access_to_owned_file(self):
        request1 = self.client2.get("/v1/data/manage/1/")
        request2 = self.client2.get("/v1/data/manage/2/")
        self.assertEqual(request1.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(request2.status_code, status.HTTP_403_FORBIDDEN)

    def test_only_edit_access_to_owned_file(self):
        data1 = {"username": "testUser2", "access": True}
        data2 = {"username": "testUser1", "access": False}
        request1 = self.client2.put("/v1/data/manage/1/", data=data1)
        request2 = self.client2.put("/v1/data/manage/2/", data=data2)
        request3 = self.client2.get("/v1/data/load/1/")
        request4 = self.client1.get("/v1/data/load/2/")
        self.assertEqual(request1.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(request2.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(request3.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(request4.status_code, status.HTTP_200_OK)

    def tearDown(self):
        shutil.rmtree(settings.MEDIA_ROOT)
