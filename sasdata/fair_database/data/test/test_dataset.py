from django.contrib.auth.models import User
from django.db.models import Max
from rest_framework.test import APIClient, APITestCase
from rest_framework import status

from data.models import DataSet

# test GET
# list datasets - include public and owned, disinclude private unowned
# get one dataset - succeeds if public or owned, fails if private and unowned
# get list of people with access to dataset (owner)
# fail to get list of people with access to dataset (not owner)

# test POST
# create a public dataset
# create a private dataset
# can't create an unowned private dataset

# test PUT
# edit an owned dataset
# can't edit an unowned dataset
# change access to a dataset

# test DELETE
# delete an owned private dataset
# can't delete an unowned dataset
# can't delete a public dataset

"""
DataSetView:
    get
    - requires 3 users, private data, public data, probably unowned data
        for authenticated user (can see public data and own private data)
        - can't see someone else's private data (may or may not be a separate test)
        for unauthenticated user (can see public data)
        with username specified
        with nonexistent username specified
    post
    - requires 2 users, no data
        for authenticated user
        for unauthenticated user, public
        for unauthenticated, private (fails)
    put
    - probably 1 user, no data
        should be same as post

SingleDataSetView:
    get
        for owned private data
        for unowned private data authenticated
        for private data unauthenticated
        for public data
    put
        same as get I think
    delete"""

# TODO: test unauthorized requests
# TODO: test permissions for users w/ access granted


class TestDataSet(APITestCase):
    @classmethod
    def setUpTestData(cls):
        cls.empty_metadata = {
            "title": "New Metadata",
            "run": "X",
            "description": "test",
            "instrument": {},
            "process": {},
            "sample": {},
            "transmission_spectrum": {},
            "raw_metadata": {},
        }
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
            metadata=None,
        )
        cls.private_dataset = DataSet.objects.create(
            id=2, current_user=cls.user1, name="Dataset 2", metadata=None
        )
        cls.unowned_dataset = DataSet.objects.create(
            id=3, is_public=True, name="Dataset 3", metadata=None
        )
        cls.private_dataset.users.add(cls.user3)
        cls.auth_client1 = APIClient()
        cls.auth_client2 = APIClient()
        cls.auth_client3 = APIClient()
        cls.auth_client1.force_authenticate(cls.user1)
        cls.auth_client2.force_authenticate(cls.user2)
        cls.auth_client3.force_authenticate(cls.user3)

    def test_list_private(self):
        request = self.auth_client1.get("/v1/data/set/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data,
            {"dataset_ids": {1: "Dataset 1", 2: "Dataset 2", 3: "Dataset 3"}},
        )

    def test_list_public(self):
        request = self.auth_client2.get("/v1/data/set/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data, {"dataset_ids": {1: "Dataset 1", 3: "Dataset 3"}}
        )

    def test_list_granted_access(self):
        request = self.auth_client3.get("/v1/data/set/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data,
            {"dataset_ids": {1: "Dataset 1", 2: "Dataset 2", 3: "Dataset 3"}},
        )

    def test_list_unauthenticated(self):
        request = self.client.get("/v1/data/set/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data, {"dataset_ids": {1: "Dataset 1", 3: "Dataset 3"}}
        )

    def test_list_username(self):
        request = self.auth_client1.get("/v1/data/set/", data={"username": "testUser1"})
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data, {"dataset_ids": {1: "Dataset 1", 2: "Dataset 2"}}
        )

    def test_list_username_2(self):
        request = self.auth_client1.get("/v1/data/set/", {"username": "testUser2"})
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(request.data, {"dataset_ids": {}})

    def test_list_username_unauthenticated(self):
        request = self.client.get("/v1/data/set/", {"username": "testUser1"})
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(request.data, {"dataset_ids": {1: "Dataset 1"}})

    def test_list_wrong_username(self):
        request = self.auth_client1.get("/v1/data/set/", {"username": "fakeUser1"})
        self.assertEqual(request.status_code, status.HTTP_404_NOT_FOUND)

    # TODO: test listing by other parameters if functionality is added for that

    def test_dataset_created(self):
        dataset = {"name": "New Dataset", "metadata": self.empty_metadata}
        request = self.auth_client1.post("/v1/data/set/", data=dataset, format="json")
        max_id = DataSet.objects.aggregate(Max("id"))["id__max"]
        new_dataset = DataSet.objects.get(id=max_id)
        new_metadata = new_dataset.metadata
        self.assertEqual(request.status_code, status.HTTP_201_CREATED)
        self.assertEqual(
            request.data,
            {"dataset_id": max_id, "name": "New Dataset", "is_public": False},
        )
        self.assertEqual(new_dataset.name, "New Dataset")
        self.assertEqual(new_metadata.title, "New Metadata")
        self.assertEqual(new_dataset.current_user.username, "testUser1")
        new_dataset.delete()
        new_metadata.delete()

    def test_dataset_created_unauthenticated(self):
        dataset = {
            "name": "New Dataset",
            "metadata": self.empty_metadata,
            "is_public": True,
        }
        request = self.client.post("/v1/data/set/", data=dataset, format="json")
        max_id = DataSet.objects.aggregate(Max("id"))["id__max"]
        new_dataset = DataSet.objects.get(id=max_id)
        new_metadata = new_dataset.metadata
        self.assertEqual(request.status_code, status.HTTP_201_CREATED)
        self.assertEqual(
            request.data,
            {"dataset_id": max_id, "name": "New Dataset", "is_public": True},
        )
        self.assertEqual(new_dataset.name, "New Dataset")
        self.assertIsNone(new_dataset.current_user)
        new_dataset.delete()
        new_metadata.delete()

    def test_no_private_unowned_dataset(self):
        dataset = {
            "name": "Disallowed Dataset",
            "metadata": self.empty_metadata,
            "is_public": False,
        }
        request = self.client.post("/v1/data/set/", data=dataset, format="json")
        self.assertEqual(request.status_code, status.HTTP_400_BAD_REQUEST)

    @classmethod
    def tearDownClass(cls):
        cls.public_dataset.delete()
        cls.private_dataset.delete()
        cls.unowned_dataset.delete()
        cls.user1.delete()
        cls.user2.delete()
        cls.user3.delete()


class TestSingleDataSet(APITestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user1 = User.objects.create_user(
            id=1, username="testUser1", password="secret"
        )
        cls.user2 = User.objects.create_user(
            id=2, username="testUser2", password="secret"
        )
        cls.public_dataset = DataSet.objects.create(
            id=1,
            current_user=cls.user1,
            is_public=True,
            name="Dataset 1",
            metadata=None,
        )
        cls.private_dataset = DataSet.objects.create(
            id=2, current_user=cls.user1, name="Dataset 2", metadata=None
        )
        cls.unowned_dataset = DataSet.objects.create(
            id=3, is_public=True, name="Dataset 3", metadata=None
        )
        cls.auth_client1 = APIClient()
        cls.auth_client2 = APIClient()
        cls.auth_client1.force_authenticate(cls.user1)
        cls.auth_client2.force_authenticate(cls.user2)

    # TODO: change load return data
    def test_load_private_dataset(self):
        request = self.auth_client1.get("/v1/data/set/2/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertDictEqual(
            request.data,
            {
                "id": 2,
                "current_user": 1,
                "users": [],
                "is_public": False,
                "name": "Dataset 2",
                "files": [],
                "metadata": None,
            },
        )

    def test_load_public_dataset(self):
        request1 = self.client.get("/v1/data/set/1/")
        request2 = self.auth_client2.get("/v1/data/set/1/")
        self.assertEqual(request1.status_code, status.HTTP_200_OK)
        self.assertEqual(request2.status_code, status.HTTP_200_OK)
        self.assertDictEqual(
            request1.data,
            {
                "id": 1,
                "current_user": 1,
                "users": [],
                "is_public": True,
                "name": "Dataset 1",
                "files": [],
                "metadata": None,
            },
        )

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
            },
        )

    def test_load_private_dataset_unauthorized(self):
        request1 = self.auth_client2.get("/v1/data/set/2/")
        request2 = self.client.get("/v1/data/set/2/")
        self.assertEqual(request1.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(request2.status_code, status.HTTP_403_FORBIDDEN)

    # TODO: check put return data
    def test_update_private_dataset(self):
        request = self.auth_client1.put("/v1/data/set/2/", data={"is_public": True})
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertTrue(DataSet.objects.get(id=2).is_public)
        self.private_dataset.save()
        self.assertFalse(DataSet.objects.get(id=2).is_public)

    def test_update_public_dataset(self):
        request = self.auth_client1.put(
            "/v1/data/set/1/", data={"name": "Different name"}
        )
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(DataSet.objects.get(id=1).name, "Different name")
        self.public_dataset.save()

    # TODO: test updating metadata

    def test_delete_dataset(self):
        request = self.auth_client1.delete("/v1/data/set/2/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertRaises(DataSet.DoesNotExist, DataSet.objects.get, id=2)
        self.private_dataset = DataSet.objects.create(
            id=2, current_user=self.user1, name="Dataset 2", metadata=None
        )

    @classmethod
    def tearDownClass(cls):
        cls.public_dataset.delete()
        cls.private_dataset.delete()
        cls.unowned_dataset.delete()
        cls.user1.delete()
        cls.user2.delete()


class TestDataSetAccessManagement(APITestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user1 = User.objects.create_user(username="testUser1", password="secret")
        cls.user2 = User.objects.create_user(username="testUser2", password="secret")
        cls.private_dataset = DataSet.objects.create(
            id=1, current_user=cls.user1, name="Dataset 1", metadata=None
        )
        cls.shared_dataset = DataSet.objects.create(
            id=2, current_user=cls.user1, name="Dataset 2", metadata=None
        )
        cls.shared_dataset.users.add(cls.user2)
        cls.client1 = APIClient()
        cls.client2 = APIClient()
        cls.client1.force_authenticate(cls.user1)
        cls.client2.force_authenticate(cls.user2)

    def test_list_access_private(self):
        request = self.client1.get("/v1/data/set/1/users/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(request.data, {"data_id": 1, "name": "Dataset 1", "users": []})

    def test_list_access_shared(self):
        request = self.client1.get("/v1/data/set/2/users/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data, {"data_id": 2, "name": "Dataset 2", "users": ["testUser2"]}
        )

    def test_list_access_unauthorized(self):
        request = self.client2.get("/v1/data/set/2/users/")
        self.assertEqual(request.status_code, status.HTTP_403_FORBIDDEN)

    def test_grant_access(self):
        request1 = self.client1.put(
            "/v1/data/set/1/users/", data={"username": "testUser2", "access": True}
        )
        request2 = self.client2.get("/v1/data/set/1/")
        self.assertEqual(request1.status_code, status.HTTP_200_OK)
        self.assertEqual(request2.status_code, status.HTTP_200_OK)
        self.assertIn(  # codespell:ignore
            self.user2, DataSet.objects.get(id=1).users.all()
        )

    @classmethod
    def tearDownClass(cls):
        cls.private_dataset.delete()
        cls.shared_dataset.delete()
        cls.user1.delete()
        cls.user2.delete()
