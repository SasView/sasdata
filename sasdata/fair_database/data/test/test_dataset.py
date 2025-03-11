from django.contrib.auth.models import User
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


class TestDataSet(APITestCase):
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

    def test_list_unauthenticated(self):
        request = self.client.get("/v1/data/set/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data, {"dataset_ids": {1: "Dataset 1", 3: "Dataset 3"}}
        )

    # TODO: test unauthorized gets

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

    # TODO: test listing by user that doesn't exist
    # TODO: test listing by other parameters if functionality is added for that

    # TODO: write test for post - probably will need to change post method to account for owner
    def test_dataset_created(self):
        pass

    def test_dataset_created_unauthenticated(self):
        pass

    def test_no_private_unowned_dataset(self):
        pass

    @classmethod
    def tearDownClass(cls):
        cls.public_dataset.delete()
        cls.private_dataset.delete()
        cls.unowned_dataset.delete()
        cls.user1.delete()
        cls.user2.delete()


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

    @classmethod
    def tearDownClass(cls):
        cls.private_dataset.delete()
        cls.shared_dataset.delete()
        cls.user1.delete()
        cls.user2.delete()
