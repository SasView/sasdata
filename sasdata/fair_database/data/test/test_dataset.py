from django.contrib.auth.models import User
from django.db.models import Max
from rest_framework.test import APIClient, APITestCase
from rest_framework import status

from data.models import DataSet, MetaData, OperationTree, Quantity


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
            {"dataset_id": max_id, "name": "New Dataset", "is_public": False},
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
            {"dataset_id": max_id, "name": "New Dataset", "is_public": True},
        )
        self.assertEqual(new_dataset.name, "New Dataset")
        self.assertIsNone(new_dataset.current_user)
        new_dataset.delete()
        new_metadata.delete()

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
            {"dataset_id": max_id, "name": "Overwrite Dataset", "is_public": True},
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

    # TODO: change load return data
    # Test successfully accessing a private dataset
    def test_load_private_dataset(self):
        request1 = self.auth_client1.get("/v1/data/set/2/")
        request2 = self.auth_client3.get("/v1/data/set/2/")
        self.assertEqual(request1.status_code, status.HTTP_200_OK)
        self.assertEqual(request2.status_code, status.HTTP_200_OK)
        self.assertDictEqual(
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
        request = self.auth_client1.delete("/v1/data/set/2/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(request.data, {"success": True})
        self.assertRaises(DataSet.DoesNotExist, DataSet.objects.get, id=2)
        self.private_dataset = DataSet.objects.create(
            id=2, current_user=self.user1, name="Dataset 2", metadata=None
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


class TestDataSetAccessManagement(APITestCase):
    """Tests for HTTP methods of DataSetUsersView."""

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
        cls.client_owner = APIClient()
        cls.client_other = APIClient()
        cls.client_owner.force_authenticate(cls.user1)
        cls.client_other.force_authenticate(cls.user2)

    # Test listing no users with access
    def test_list_access_private(self):
        request1 = self.client_owner.get("/v1/data/set/1/users/")
        self.assertEqual(request1.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request1.data, {"data_id": 1, "name": "Dataset 1", "users": []}
        )

    # Test listing users with access
    def test_list_access_shared(self):
        request1 = self.client_owner.get("/v1/data/set/2/users/")
        self.assertEqual(request1.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request1.data, {"data_id": 2, "name": "Dataset 2", "users": ["testUser2"]}
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


class TestOperationTree(APITestCase):
    """Tests for datasets with operation trees."""

    @classmethod
    def setUpTestData(cls):
        cls.dataset = {
            "name": "Test Dataset",
            "metadata": {
                "title": "test metadata",
                "run": 1,
                "definition": "test",
                "instrument": {"source": {}, "collimation": {}, "detectors": {}},
            },
            "data_contents": [
                {
                    "label": "test",
                    "value": {"array_contents": [0, 0, 0, 0], "shape": (2, 2)},
                    "variance": {"array_contents": [0, 0, 0, 0], "shape": (2, 2)},
                    "units": "none",
                    "hash": 0,
                }
            ],
            "is_public": True,
        }
        cls.nested_operations = {
            "operation_tree": {
                "operation": "neg",
                "parameters": {
                    "a": {
                        "operation": "mul",
                        "parameters": {
                            "a": {
                                "operation": "constant",
                                "parameters": {"value": {"type": "int", "value": 7}},
                            },
                            "b": {
                                "operation": "variable",
                                "parameters": {"hash_value": 111, "name": "x"},
                            },
                        },
                    },
                },
            },
            "references": {},
        }
        cls.user = User.objects.create_user(
            id=1, username="testUser", password="sasview!"
        )
        cls.client = APIClient()
        cls.client.force_authenticate(cls.user)

    # Test creating quantity with no operations performed
    def test_operation_tree_created_variable(self):
        self.dataset["data_contents"][0]["history"] = {
            "operation_tree": {
                "operation": "variable",
                "parameters": {"hash_value": 0, "name": "test"},
            },
            "references": {},
        }
        request = self.client.post("/v1/data/set/", data=self.dataset, format="json")
        max_id = DataSet.objects.aggregate(Max("id"))["id__max"]
        new_dataset = DataSet.objects.get(id=max_id)
        new_quantity = new_dataset.data_contents.get(hash=0)
        self.assertEqual(request.status_code, status.HTTP_201_CREATED)
        self.assertIsNone(new_quantity.operation_tree)

    # Test accessing a quantity with no operations performed

    # Test creating quantity with unary operation
    def test_operation_tree_created_unary(self):
        self.dataset["data_contents"][0]["history"] = {
            "operation_tree": {
                "operation": "reciprocal",
                "parameters": {
                    "a": {
                        "operation": "variable",
                        "parameters": {"hash_value": 111, "name": "x"},
                    }
                },
            },
            "references": {},
        }
        request = self.client.post("/v1/data/set/", data=self.dataset, format="json")
        max_id = DataSet.objects.aggregate(Max("id"))["id__max"]
        new_dataset = DataSet.objects.get(id=max_id)
        new_quantity = new_dataset.data_contents.get(hash=0)
        reciprocal = new_quantity.operation_tree
        self.assertEqual(request.status_code, status.HTTP_201_CREATED)
        self.assertEqual(
            new_quantity.value, {"array_contents": [0, 0, 0, 0], "shape": [2, 2]}
        )
        self.assertEqual(reciprocal.operation, "reciprocal")
        self.assertEqual(reciprocal.parent_operation1.operation, "variable")
        self.assertEqual(reciprocal.parameters, {})

    # Test accessing quantity with unary operation
    def test_get_operation_tree_unary(self):
        variable = OperationTree.objects.create(
            id=1, operation="variable", parameters={"hash_value": 111, "name": "x"}
        )
        inv = OperationTree.objects.create(
            id=2, operation="reciprocal", parent_operation1=variable
        )
        quantity = Quantity.objects.create(
            id=1,
            value=0,
            variance=0,
            label="test",
            units="none",
            hash=1,
            operation_tree=inv,
        )
        dataset = DataSet.objects.create(
            id=1,
            current_user=self.user,
            name="Test Dataset",
            is_public=True,
            metadata=None,
        )
        dataset.data_contents.add(quantity)
        request = self.client.get("/v1/data/set/1/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data,
            {
                "id": 1,
                "current_user": 1,
                "users": [],
                "is_public": True,
                "name": "Test Dataset",
                "files": [],
                "metadata": None,
                "data_contents": [
                    {
                        "label": "test",
                        "value": 0,
                        "variance": 0,
                        "units": "none",
                        "hash": 1,
                        "operation_tree": {
                            "operation": "reciprocal",
                            "parameters": {
                                "a": {
                                    "operation": "variable",
                                    "parameters": {"hash_value": 111, "name": "x"},
                                }
                            },
                        },
                    }
                ],
            },
        )

    # Test creating quantity with binary operation
    def test_operation_tree_created_binary(self):
        self.dataset["data_contents"][0]["history"] = {
            "operation_tree": {
                "operation": "add",
                "parameters": {
                    "a": {
                        "operation": "variable",
                        "parameters": {"hash_value": 111, "name": "x"},
                    },
                    "b": {"operation": "constant", "parameters": {"value": 5}},
                },
            },
            "references": {},
        }
        request = self.client.post("/v1/data/set/", data=self.dataset, format="json")
        max_id = DataSet.objects.aggregate(Max("id"))["id__max"]
        new_dataset = DataSet.objects.get(id=max_id)
        new_quantity = new_dataset.data_contents.get(hash=0)
        add = new_quantity.operation_tree
        variable = add.parent_operation1
        constant = add.parent_operation2
        self.assertEqual(request.status_code, status.HTTP_201_CREATED)
        self.assertEqual(add.operation, "add")
        self.assertEqual(add.parameters, {})
        self.assertEqual(variable.operation, "variable")
        self.assertEqual(variable.parameters, {"hash_value": 111, "name": "x"})
        self.assertEqual(constant.operation, "constant")
        self.assertEqual(constant.parameters, {"value": 5})

    # Test accessing quantity with binary operation
    def test_get_operation_tree_binary(self):
        variable = OperationTree.objects.create(
            id=1, operation="variable", parameters={"hash_value": 111, "name": "x"}
        )
        constant = OperationTree.objects.create(
            id=2, operation="constant", parameters={"value": 1}
        )
        add = OperationTree.objects.create(
            id=3,
            operation="add",
            parent_operation1=variable,
            parent_operation2=constant,
        )
        quantity = Quantity.objects.create(
            id=1,
            value=0,
            variance=0,
            label="test",
            units="none",
            hash=1,
            operation_tree=add,
        )
        dataset = DataSet.objects.create(
            id=1,
            current_user=self.user,
            name="Test Dataset",
            is_public=True,
            metadata=None,
        )
        dataset.data_contents.add(quantity)
        request = self.client.get("/v1/data/set/1/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data,
            {
                "id": 1,
                "current_user": 1,
                "users": [],
                "is_public": True,
                "name": "Test Dataset",
                "files": [],
                "metadata": None,
                "data_contents": [
                    {
                        "label": "test",
                        "value": 0,
                        "variance": 0,
                        "units": "none",
                        "hash": 1,
                        "operation_tree": {
                            "operation": "add",
                            "parameters": {
                                "a": {
                                    "operation": "variable",
                                    "parameters": {"hash_value": 111, "name": "x"},
                                },
                                "b": {
                                    "operation": "constant",
                                    "parameters": {"value": 1},
                                },
                            },
                        },
                    }
                ],
            },
        )

    # Test creating quantity with exponent
    def test_operation_tree_created_pow(self):
        self.dataset["data_contents"][0]["history"] = {
            "operation_tree": {
                "operation": "pow",
                "parameters": {
                    "a": {
                        "operation": "variable",
                        "parameters": {"hash_value": 111, "name": "x"},
                    },
                    "power": 2,
                },
            },
            "references": {},
        }
        request = self.client.post("/v1/data/set/", data=self.dataset, format="json")
        max_id = DataSet.objects.aggregate(Max("id"))["id__max"]
        new_dataset = DataSet.objects.get(id=max_id)
        new_quantity = new_dataset.data_contents.get(hash=0)
        pow = new_quantity.operation_tree
        self.assertEqual(request.status_code, status.HTTP_201_CREATED)
        self.assertEqual(pow.operation, "pow")
        self.assertEqual(pow.parameters, {"power": 2})

    # Test accessing a quantity with exponent
    def test_get_operation_tree_pow(self):
        variable = OperationTree.objects.create(
            id=1, operation="variable", parameters={"hash_value": 111, "name": "x"}
        )
        pow = OperationTree.objects.create(
            id=2, operation="pow", parent_operation1=variable, parameters={"power": 2}
        )
        quantity = Quantity.objects.create(
            id=1,
            value=0,
            variance=0,
            label="test",
            units="none",
            hash=1,
            operation_tree=pow,
        )
        dataset = DataSet.objects.create(
            id=1,
            current_user=self.user,
            name="Test Dataset",
            is_public=True,
            metadata=None,
        )
        dataset.data_contents.add(quantity)
        request = self.client.get("/v1/data/set/1/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data,
            {
                "id": 1,
                "current_user": 1,
                "users": [],
                "is_public": True,
                "name": "Test Dataset",
                "files": [],
                "metadata": None,
                "data_contents": [
                    {
                        "label": "test",
                        "value": 0,
                        "variance": 0,
                        "units": "none",
                        "hash": 1,
                        "operation_tree": {
                            "operation": "pow",
                            "parameters": {
                                "a": {
                                    "operation": "variable",
                                    "parameters": {"hash_value": 111, "name": "x"},
                                },
                                "power": 2,
                            },
                        },
                    }
                ],
            },
        )

    # Test creating a transposed quantity
    def test_operation_tree_created_transpose(self):
        self.dataset["data_contents"][0]["history"] = {
            "operation_tree": {
                "operation": "transpose",
                "parameters": {
                    "a": {
                        "operation": "variable",
                        "parameters": {"hash_value": 111, "name": "x"},
                    },
                    "axes": [1, 0],
                },
            },
            "references": {},
        }
        request = self.client.post("/v1/data/set/", data=self.dataset, format="json")
        max_id = DataSet.objects.aggregate(Max("id"))["id__max"]
        new_dataset = DataSet.objects.get(id=max_id)
        new_quantity = new_dataset.data_contents.get(hash=0)
        transpose = new_quantity.operation_tree
        variable = transpose.parent_operation1
        self.assertEqual(request.status_code, status.HTTP_201_CREATED)
        self.assertEqual(transpose.operation, "transpose")
        self.assertEqual(transpose.parameters, {"axes": [1, 0]})
        self.assertEqual(variable.operation, "variable")
        self.assertEqual(variable.parameters, {"hash_value": 111, "name": "x"})

    # Test accessing a transposed quantity
    def test_get_operation_tree_transpose(self):
        variable = OperationTree.objects.create(
            id=1, operation="variable", parameters={"hash_value": 111, "name": "x"}
        )
        inv = OperationTree.objects.create(
            id=2,
            operation="transpose",
            parent_operation1=variable,
            parameters={"axes": (1, 0)},
        )
        quantity = Quantity.objects.create(
            id=1,
            value=0,
            variance=0,
            label="test",
            units="none",
            hash=1,
            operation_tree=inv,
        )
        dataset = DataSet.objects.create(
            id=1,
            current_user=self.user,
            name="Test Dataset",
            is_public=True,
            metadata=None,
        )
        dataset.data_contents.add(quantity)
        request = self.client.get("/v1/data/set/1/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data,
            {
                "id": 1,
                "current_user": 1,
                "users": [],
                "is_public": True,
                "name": "Test Dataset",
                "files": [],
                "metadata": None,
                "data_contents": [
                    {
                        "label": "test",
                        "value": 0,
                        "variance": 0,
                        "units": "none",
                        "hash": 1,
                        "operation_tree": {
                            "operation": "transpose",
                            "parameters": {
                                "a": {
                                    "operation": "variable",
                                    "parameters": {"hash_value": 111, "name": "x"},
                                },
                                "axes": [1, 0],
                            },
                        },
                    }
                ],
            },
        )

    # Test creating a quantity with multiple operations
    def test_operation_tree_created_nested(self):
        self.dataset["data_contents"][0]["history"] = self.nested_operations
        request = self.client.post("/v1/data/set/", data=self.dataset, format="json")
        max_id = DataSet.objects.aggregate(Max("id"))["id__max"]
        new_dataset = DataSet.objects.get(id=max_id)
        new_quantity = new_dataset.data_contents.get(hash=0)
        negate = new_quantity.operation_tree
        multiply = negate.parent_operation1
        constant = multiply.parent_operation1
        variable = multiply.parent_operation2
        self.assertEqual(request.status_code, status.HTTP_201_CREATED)
        self.assertEqual(negate.operation, "neg")
        self.assertEqual(negate.parameters, {})
        self.assertEqual(multiply.operation, "mul")
        self.assertEqual(multiply.parameters, {})
        self.assertEqual(constant.operation, "constant")
        self.assertEqual(constant.parameters, {"value": {"type": "int", "value": 7}})
        self.assertEqual(variable.operation, "variable")
        self.assertEqual(variable.parameters, {"hash_value": 111, "name": "x"})

    # Test accessing a quantity with multiple operations
    def test_get_operation_tree_nested(self):
        variable = OperationTree.objects.create(
            id=1, operation="variable", parameters={"hash_value": 111, "name": "x"}
        )
        constant = OperationTree.objects.create(
            id=2,
            operation="constant",
            parameters={"value": {"type": "int", "value": 7}},
        )
        multiply = OperationTree.objects.create(
            id=3,
            operation="mul",
            parent_operation1=constant,
            parent_operation2=variable,
        )
        neg = OperationTree.objects.create(
            id=4, operation="neg", parent_operation1=multiply
        )
        quantity = Quantity.objects.create(
            id=1,
            value=0,
            variance=0,
            label="test",
            units="none",
            hash=1,
            operation_tree=neg,
        )
        dataset = DataSet.objects.create(
            id=1,
            current_user=self.user,
            name="Test Dataset",
            is_public=True,
            metadata=None,
        )
        dataset.data_contents.add(quantity)
        request = self.client.get("/v1/data/set/1/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data,
            {
                "id": 1,
                "current_user": 1,
                "users": [],
                "is_public": True,
                "name": "Test Dataset",
                "files": [],
                "metadata": None,
                "data_contents": [
                    {
                        "label": "test",
                        "value": 0,
                        "variance": 0,
                        "units": "none",
                        "hash": 1,
                        "operation_tree": {
                            "operation": "neg",
                            "parameters": {
                                "a": {
                                    "operation": "mul",
                                    "parameters": {
                                        "a": {
                                            "operation": "constant",
                                            "parameters": {
                                                "value": {"type": "int", "value": 7}
                                            },
                                        },
                                        "b": {
                                            "operation": "variable",
                                            "parameters": {
                                                "hash_value": 111,
                                                "name": "x",
                                            },
                                        },
                                    },
                                }
                            },
                        },
                    }
                ],
            },
        )

    # Test creating a quantity with tensordot

    # Test accessing a quantity with tensordot

    # Test creating a quantity with an invalid operation
    def test_create_operation_tree_invalid(self):
        self.dataset["data_contents"][0]["history"] = {
            "operation_tree": {"operation": "fix", "parameters": {}},
            "references": {},
        }
        request = self.client.post("/v1/data/set/", data=self.dataset, format="json")
        self.assertEqual(request.status_code, status.HTTP_400_BAD_REQUEST)

    # Test creating a quantity with a nested invalid operation

    # Test creating invalid operation parameters
    # binary has a and b - both should be operations
    # unary has a (operation)
    # constant has value
    # variable has name and hash_value
    # pow has power
    # transpose has axes
    # tensordot has a_index and b_index

    # Test creating nested invalid operation parameters

    # Test creating a quantity with no history

    def tearDown(self):
        DataSet.objects.all().delete()
        MetaData.objects.all().delete()
        Quantity.objects.all().delete()
        OperationTree.objects.all().delete()

    @classmethod
    def tearDownClass(cls):
        cls.user.delete()
