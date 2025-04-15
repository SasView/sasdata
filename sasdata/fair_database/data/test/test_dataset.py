import os
import shutil

from django.conf import settings
from django.contrib.auth.models import User
from django.db.models import Max
from rest_framework.test import APIClient, APITestCase
from rest_framework import status

from data.models import DataFile, DataSet, MetaData, OperationTree, Quantity


def find(filename):
    return os.path.join(
        os.path.dirname(__file__), "../../../example_data/1d_data", filename
    )


class TestDataSet(APITestCase):
    """Test HTTP methods of DataSetView."""

    @classmethod
    def setUpTestData(cls):
        cls.empty_metadata = {
            "title": "New Metadata",
            "run": ["X"],
            "description": "test",
            "instrument": {},
            "process": {},
            "sample": {},
        }
        cls.empty_data = [
            {"value": 0, "variance": 0, "units": "no", "hash": 0, "label": "test"}
        ]
        cls.user1 = User.objects.create_user(
            id=1, username="testUser1", password="secret"
        )
        cls.user2 = User.objects.create_user(
            id=2, username="testUser2", password="secret"
        )
        cls.user3 = User.objects.create_user(
            id=3, username="testUser3", password="secret"
        )
        cls.public_dataset = DataSet.objects.create(
            id=1,
            current_user=cls.user1,
            is_public=True,
            name="Dataset 1",
        )
        cls.private_dataset = DataSet.objects.create(
            id=2, current_user=cls.user1, name="Dataset 2"
        )
        cls.unowned_dataset = DataSet.objects.create(
            id=3, is_public=True, name="Dataset 3"
        )
        cls.private_dataset.users.add(cls.user3)
        cls.auth_client1 = APIClient()
        cls.auth_client2 = APIClient()
        cls.auth_client3 = APIClient()
        cls.auth_client1.force_authenticate(cls.user1)
        cls.auth_client2.force_authenticate(cls.user2)
        cls.auth_client3.force_authenticate(cls.user3)

    # Test a user can list their own private data
    def test_list_private(self):
        request = self.auth_client1.get("/v1/data/set/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data,
            {"dataset_ids": {1: "Dataset 1", 2: "Dataset 2", 3: "Dataset 3"}},
        )

    # Test a user can see others' public but not private data in list
    def test_list_public(self):
        request = self.auth_client2.get("/v1/data/set/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data, {"dataset_ids": {1: "Dataset 1", 3: "Dataset 3"}}
        )

    # Test a user can see private data they have been granted access to
    def test_list_granted_access(self):
        request = self.auth_client3.get("/v1/data/set/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data,
            {"dataset_ids": {1: "Dataset 1", 2: "Dataset 2", 3: "Dataset 3"}},
        )

    # Test an unauthenticated user can list public data
    def test_list_unauthenticated(self):
        request = self.client.get("/v1/data/set/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data, {"dataset_ids": {1: "Dataset 1", 3: "Dataset 3"}}
        )

    # Test a user can see all data listed by their username
    def test_list_username(self):
        request = self.auth_client1.get("/v1/data/set/", data={"username": "testUser1"})
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data, {"dataset_ids": {1: "Dataset 1", 2: "Dataset 2"}}
        )

    # Test a user can list public data by another user's username
    def test_list_username_2(self):
        request = self.auth_client1.get("/v1/data/set/", {"username": "testUser2"})
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(request.data, {"dataset_ids": {}})

    # Test an unauthenticated user can list public data by a username
    def test_list_username_unauthenticated(self):
        request = self.client.get("/v1/data/set/", {"username": "testUser1"})
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(request.data, {"dataset_ids": {1: "Dataset 1"}})

    # Test listing by a username that doesn't exist
    def test_list_wrong_username(self):
        request = self.auth_client1.get("/v1/data/set/", {"username": "fakeUser1"})
        self.assertEqual(request.status_code, status.HTTP_404_NOT_FOUND)

    # TODO: test listing by other parameters if functionality is added for that

    # Test creating a dataset with associated metadata
    def test_dataset_created(self):
        dataset = {
            "name": "New Dataset",
            "metadata": self.empty_metadata,
            "data_contents": self.empty_data,
        }
        request = self.auth_client1.post("/v1/data/set/", data=dataset, format="json")
        max_id = DataSet.objects.aggregate(Max("id"))["id__max"]
        new_dataset = DataSet.objects.get(id=max_id)
        new_metadata = new_dataset.metadata
        self.assertEqual(request.status_code, status.HTTP_201_CREATED)
        self.assertEqual(
            request.data,
            {
                "dataset_id": max_id,
                "name": "New Dataset",
                "authenticated": True,
                "current_user": "testUser1",
                "is_public": False,
            },
        )
        self.assertEqual(new_dataset.name, "New Dataset")
        self.assertEqual(new_metadata.title, "New Metadata")
        self.assertEqual(new_dataset.current_user.username, "testUser1")
        new_dataset.delete()
        new_metadata.delete()

    # Test creating a dataset while unauthenticated
    def test_dataset_created_unauthenticated(self):
        dataset = {
            "name": "New Dataset",
            "metadata": self.empty_metadata,
            "is_public": True,
            "data_contents": self.empty_data,
        }
        request = self.client.post("/v1/data/set/", data=dataset, format="json")
        max_id = DataSet.objects.aggregate(Max("id"))["id__max"]
        new_dataset = DataSet.objects.get(id=max_id)
        new_metadata = new_dataset.metadata
        self.assertEqual(request.status_code, status.HTTP_201_CREATED)
        self.assertEqual(
            request.data,
            {
                "dataset_id": max_id,
                "name": "New Dataset",
                "authenticated": False,
                "current_user": "",
                "is_public": True,
            },
        )
        self.assertEqual(new_dataset.name, "New Dataset")
        self.assertIsNone(new_dataset.current_user)
        new_dataset.delete()
        new_metadata.delete()

    # Test creating a database with associated files
    def test_dataset_created_with_files(self):
        file = DataFile.objects.create(
            id=1, file_name="cyl_testdata.txt", is_public=True
        )
        file.file.save("cyl_testdata.txt", open(find("cyl_testdata.txt")))
        dataset = {
            "name": "Dataset with file",
            "metadata": self.empty_metadata,
            "is_public": True,
            "data_contents": self.empty_data,
            "files": [1],
        }
        request = self.client.post("/v1/data/set/", data=dataset, format="json")
        max_id = DataSet.objects.aggregate(Max("id"))["id__max"]
        new_dataset = DataSet.objects.get(id=max_id)
        self.assertEqual(request.status_code, status.HTTP_201_CREATED)
        self.assertEqual(
            request.data,
            {
                "dataset_id": max_id,
                "name": "Dataset with file",
                "authenticated": False,
                "current_user": "",
                "is_public": True,
            },
        )
        self.assertTrue(file in new_dataset.files.all())
        new_dataset.delete()
        file.delete()

    # Test that a dataset cannot be associated with inaccessible files
    def test_no_dataset_with_private_files(self):
        file = DataFile.objects.create(
            id=1, file_name="cyl_testdata.txt", is_public=False, current_user=self.user2
        )
        file.file.save("cyl_testdata.txt", open(find("cyl_testdata.txt")))
        dataset = {
            "name": "Dataset with file",
            "metadata": self.empty_metadata,
            "is_public": True,
            "data_contents": self.empty_data,
            "files": [1],
        }
        request = self.client.post("/v1/data/set/", data=dataset, format="json")
        file.delete()
        self.assertEqual(request.status_code, status.HTTP_400_BAD_REQUEST)

    # Test that a dataset cannot be associated with nonexistent files
    def test_no_dataset_with_nonexistent_files(self):
        dataset = {
            "name": "Dataset with file",
            "metadata": self.empty_metadata,
            "is_public": True,
            "data_contents": self.empty_data,
            "files": [2],
        }
        request = self.client.post("/v1/data/set/", data=dataset, format="json")
        self.assertEqual(request.status_code, status.HTTP_400_BAD_REQUEST)

    # Test that a private dataset cannot be created without an owner
    def test_no_private_unowned_dataset(self):
        dataset = {
            "name": "Disallowed Dataset",
            "metadata": self.empty_metadata,
            "is_public": False,
            "data_contents": self.empty_data,
        }
        request = self.client.post("/v1/data/set/", data=dataset, format="json")
        self.assertEqual(request.status_code, status.HTTP_400_BAD_REQUEST)

    # Test whether a user can overwrite data by specifying an in-use id
    def test_no_data_overwrite(self):
        dataset = {
            "id": 2,
            "name": "Overwrite Dataset",
            "metadata": self.empty_metadata,
            "is_public": True,
            "data_contents": self.empty_data,
        }
        request = self.auth_client2.post("/v1/data/set/", data=dataset, format="json")
        max_id = DataSet.objects.aggregate(Max("id"))["id__max"]
        self.assertEqual(request.status_code, status.HTTP_201_CREATED)
        self.assertEqual(DataSet.objects.get(id=2).name, "Dataset 2")
        self.assertEqual(
            request.data,
            {
                "dataset_id": max_id,
                "name": "Overwrite Dataset",
                "authenticated": True,
                "current_user": "testUser2",
                "is_public": True,
            },
        )
        DataSet.objects.get(id=max_id).delete()

    @classmethod
    def tearDownClass(cls):
        cls.public_dataset.delete()
        cls.private_dataset.delete()
        cls.unowned_dataset.delete()
        cls.user1.delete()
        cls.user2.delete()
        cls.user3.delete()
        shutil.rmtree(settings.MEDIA_ROOT)


class TestSingleDataSet(APITestCase):
    """Tests for HTTP methods of SingleDataSetView."""

    @classmethod
    def setUpTestData(cls):
        cls.user1 = User.objects.create_user(
            id=1, username="testUser1", password="secret"
        )
        cls.user2 = User.objects.create_user(
            id=2, username="testUser2", password="secret"
        )
        cls.user3 = User.objects.create_user(
            id=3, username="testUser3", password="secret"
        )
        cls.public_dataset = DataSet.objects.create(
            id=1,
            current_user=cls.user1,
            is_public=True,
            name="Dataset 1",
        )
        cls.private_dataset = DataSet.objects.create(
            id=2, current_user=cls.user1, name="Dataset 2"
        )
        cls.unowned_dataset = DataSet.objects.create(
            id=3, is_public=True, name="Dataset 3"
        )
        cls.metadata = MetaData.objects.create(
            id=1,
            title="Metadata",
            run=0,
            definition="test",
            instrument="none",
            process="none",
            sample="none",
            dataset=cls.public_dataset,
        )
        cls.file = DataFile.objects.create(
            id=1, file_name="cyl_testdata.txt", is_public=False, current_user=cls.user1
        )
        cls.file.file.save("cyl_testdata.txt", open(find("cyl_testdata.txt")))
        cls.private_dataset.users.add(cls.user3)
        cls.public_dataset.files.add(cls.file)
        cls.auth_client1 = APIClient()
        cls.auth_client2 = APIClient()
        cls.auth_client3 = APIClient()
        cls.auth_client1.force_authenticate(cls.user1)
        cls.auth_client2.force_authenticate(cls.user2)
        cls.auth_client3.force_authenticate(cls.user3)

    # TODO: change load return data
    # Test successfully accessing a private dataset
    def test_load_private_dataset(self):
        request1 = self.auth_client1.get("/v1/data/set/2/")
        request2 = self.auth_client3.get("/v1/data/set/2/")
        self.assertEqual(request1.status_code, status.HTTP_200_OK)
        self.assertEqual(request2.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request1.data,
            {
                "id": 2,
                "current_user": 1,
                "users": [3],
                "is_public": False,
                "name": "Dataset 2",
                "files": [],
                "metadata": None,
                "data_contents": [],
            },
        )

    # Test successfully accessing a public dataset
    def test_load_public_dataset(self):
        request1 = self.client.get("/v1/data/set/1/")
        request2 = self.auth_client2.get("/v1/data/set/1/")
        request3 = self.auth_client1.get("/v1/data/set/1/")
        self.assertEqual(request1.status_code, status.HTTP_200_OK)
        self.assertEqual(request2.status_code, status.HTTP_200_OK)
        self.assertEqual(request3.status_code, status.HTTP_200_OK)
        self.assertDictEqual(
            request1.data,
            {
                "id": 1,
                "current_user": 1,
                "users": [],
                "is_public": True,
                "name": "Dataset 1",
                "files": [],
                "metadata": {
                    "id": 1,
                    "title": "Metadata",
                    "run": 0,
                    "definition": "test",
                    "instrument": "none",
                    "process": "none",
                    "sample": "none",
                },
                "data_contents": [],
            },
        )
        self.assertEqual(request1.data, request2.data)
        self.assertEqual(
            request3.data,
            {
                "id": 1,
                "current_user": 1,
                "users": [],
                "is_public": True,
                "name": "Dataset 1",
                "files": [1],
                "metadata": {
                    "id": 1,
                    "title": "Metadata",
                    "run": 0,
                    "definition": "test",
                    "instrument": "none",
                    "process": "none",
                    "sample": "none",
                },
                "data_contents": [],
            },
        )

    # Test successfully accessing an unowned public dataset
    def test_load_unowned_dataset(self):
        request1 = self.auth_client1.get("/v1/data/set/3/")
        request2 = self.client.get("/v1/data/set/3/")
        self.assertEqual(request1.status_code, status.HTTP_200_OK)
        self.assertEqual(request2.status_code, status.HTTP_200_OK)
        self.assertDictEqual(
            request1.data,
            {
                "id": 3,
                "current_user": None,
                "users": [],
                "is_public": True,
                "name": "Dataset 3",
                "files": [],
                "metadata": None,
                "data_contents": [],
            },
        )

    # Test unsuccessfully accessing a private dataset
    def test_load_private_dataset_unauthorized(self):
        request1 = self.auth_client2.get("/v1/data/set/2/")
        request2 = self.client.get("/v1/data/set/2/")
        self.assertEqual(request1.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(request2.status_code, status.HTTP_401_UNAUTHORIZED)

    # Test only owner can change a private dataset
    def test_update_private_dataset(self):
        request1 = self.auth_client1.put("/v1/data/set/2/", data={"is_public": True})
        request2 = self.auth_client3.put("/v1/data/set/2/", data={"is_public": False})
        self.assertEqual(request1.status_code, status.HTTP_200_OK)
        self.assertEqual(request2.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(
            request1.data, {"data_id": 2, "name": "Dataset 2", "is_public": True}
        )
        self.assertTrue(DataSet.objects.get(id=2).is_public)
        self.private_dataset.save()
        self.assertFalse(DataSet.objects.get(id=2).is_public)

    # Test changing a public dataset
    def test_update_public_dataset(self):
        request1 = self.auth_client1.put(
            "/v1/data/set/1/", data={"name": "Different name"}
        )
        request2 = self.auth_client2.put("/v1/data/set/1/", data={"is_public": False})
        self.assertEqual(request1.status_code, status.HTTP_200_OK)
        self.assertEqual(request2.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(
            request1.data, {"data_id": 1, "name": "Different name", "is_public": True}
        )
        self.assertEqual(DataSet.objects.get(id=1).name, "Different name")
        self.public_dataset.save()

    # TODO: test updating metadata once metadata is figured out
    # TODO: test invalid updates if and when those are figured out

    # Test changing an unowned dataset
    def test_update_unowned_dataset(self):
        request1 = self.auth_client1.put("/v1/data/set/3/", data={"current_user": 1})
        request2 = self.client.put("/v1/data/set/3/", data={"name": "Different name"})
        self.assertEqual(request1.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(request2.status_code, status.HTTP_401_UNAUTHORIZED)

    # Test deleting a dataset
    def test_delete_dataset(self):
        quantity = Quantity.objects.create(
            id=1,
            value=0,
            variance=0,
            units="none",
            hash=0,
            label="test",
            dataset=self.private_dataset,
        )
        neg = OperationTree.objects.create(id=1, operation="neg", quantity=quantity)
        OperationTree.objects.create(
            id=2, operation="zero", parameters={}, child_operation=neg
        )
        request = self.auth_client1.delete("/v1/data/set/2/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(request.data, {"success": True})
        self.assertRaises(DataSet.DoesNotExist, DataSet.objects.get, id=2)
        self.assertRaises(Quantity.DoesNotExist, Quantity.objects.get, id=1)
        self.assertRaises(OperationTree.DoesNotExist, OperationTree.objects.get, id=1)
        self.assertRaises(OperationTree.DoesNotExist, OperationTree.objects.get, id=2)
        self.private_dataset = DataSet.objects.create(
            id=2, current_user=self.user1, name="Dataset 2"
        )

    # Test cannot delete a public dataset
    def test_delete_public_dataset(self):
        request = self.auth_client1.delete("/v1/data/set/1/")
        self.assertEqual(request.status_code, status.HTTP_403_FORBIDDEN)

    # Test cannot delete an unowned dataset
    def test_delete_unowned_dataset(self):
        request = self.auth_client1.delete("/v1/data/set/3/")
        self.assertEqual(request.status_code, status.HTTP_403_FORBIDDEN)

    # Test cannot delete another user's dataset
    def test_delete_dataset_unauthorized(self):
        request1 = self.auth_client2.delete("/v1/data/set/1/")
        request2 = self.auth_client3.delete("/v1/data/set/2/")
        self.assertEqual(request1.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(request2.status_code, status.HTTP_403_FORBIDDEN)

    @classmethod
    def tearDownClass(cls):
        cls.public_dataset.delete()
        cls.private_dataset.delete()
        cls.unowned_dataset.delete()
        cls.user1.delete()
        cls.user2.delete()
        cls.user3.delete()
        cls.file.delete()
        shutil.rmtree(settings.MEDIA_ROOT)


class TestDataSetAccessManagement(APITestCase):
    """Tests for HTTP methods of DataSetUsersView."""

    @classmethod
    def setUpTestData(cls):
        cls.user1 = User.objects.create_user(username="testUser1", password="secret")
        cls.user2 = User.objects.create_user(username="testUser2", password="secret")
        cls.private_dataset = DataSet.objects.create(
            id=1, current_user=cls.user1, name="Dataset 1"
        )
        cls.shared_dataset = DataSet.objects.create(
            id=2, current_user=cls.user1, name="Dataset 2"
        )
        cls.shared_dataset.users.add(cls.user2)
        cls.client_owner = APIClient()
        cls.client_other = APIClient()
        cls.client_owner.force_authenticate(cls.user1)
        cls.client_other.force_authenticate(cls.user2)

    # Test listing no users with access
    def test_list_access_private(self):
        request1 = self.client_owner.get("/v1/data/set/1/users/")
        self.assertEqual(request1.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request1.data,
            {"data_id": 1, "name": "Dataset 1", "is_public": False, "users": []},
        )

    # Test listing users with access
    def test_list_access_shared(self):
        request1 = self.client_owner.get("/v1/data/set/2/users/")
        self.assertEqual(request1.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request1.data,
            {
                "data_id": 2,
                "name": "Dataset 2",
                "is_public": False,
                "users": ["testUser2"],
            },
        )

    # Test only owner can view access
    def test_list_access_unauthorized(self):
        request = self.client_other.get("/v1/data/set/2/users/")
        self.assertEqual(request.status_code, status.HTTP_403_FORBIDDEN)

    # Test granting access to a dataset
    def test_grant_access(self):
        request1 = self.client_owner.put(
            "/v1/data/set/1/users/", data={"username": "testUser2", "access": True}
        )
        request2 = self.client_other.get("/v1/data/set/1/")
        self.assertEqual(request1.status_code, status.HTTP_200_OK)
        self.assertEqual(request2.status_code, status.HTTP_200_OK)
        self.assertIn(  # codespell:ignore
            self.user2, DataSet.objects.get(id=1).users.all()
        )
        self.assertEqual(
            request1.data,
            {
                "username": "testUser2",
                "data_id": 1,
                "name": "Dataset 1",
                "access": True,
            },
        )
        self.private_dataset.users.remove(self.user2)

    # Test revoking access to a dataset
    def test_revoke_access(self):
        request1 = self.client_owner.put(
            "/v1/data/set/2/users/", data={"username": "testUser2", "access": False}
        )
        request2 = self.client_other.get("/v1/data/set/2/")
        self.assertEqual(request1.status_code, status.HTTP_200_OK)
        self.assertEqual(request2.status_code, status.HTTP_403_FORBIDDEN)
        self.assertNotIn(self.user2, DataSet.objects.get(id=2).users.all())
        self.assertEqual(
            request1.data,
            {
                "username": "testUser2",
                "data_id": 2,
                "name": "Dataset 2",
                "access": False,
            },
        )
        self.shared_dataset.users.add(self.user2)

    # Test only the owner can change access
    def test_revoke_access_unauthorized(self):
        request1 = self.client_other.put(
            "/v1/data/set/2/users/", data={"username": "testUser2", "access": False}
        )
        self.assertEqual(request1.status_code, status.HTTP_403_FORBIDDEN)

    @classmethod
    def tearDownClass(cls):
        cls.private_dataset.delete()
        cls.shared_dataset.delete()
        cls.user1.delete()
        cls.user2.delete()
