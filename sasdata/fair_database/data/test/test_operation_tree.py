from django.contrib.auth.models import User
from django.db.models import Max
from rest_framework.test import APIClient, APITestCase
from rest_framework import status

from data.models import DataSet, MetaData, OperationTree, Quantity


class TestCreateOperationTree(APITestCase):
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

    # Test creating a quantity with tensordot

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


class TestGetOperationTree(APITestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = User.objects.create_user(
            id=1, username="testUser", password="sasview!"
        )
        cls.quantity = Quantity.objects.create(
            id=1,
            value=0,
            variance=0,
            label="test",
            units="none",
            hash=1,
        )
        cls.dataset = DataSet.objects.create(
            id=1,
            current_user=cls.user,
            name="Test Dataset",
            is_public=True,
            metadata=None,
        )
        cls.variable = OperationTree.objects.create(
            id=1, operation="variable", parameters={"hash_value": 111, "name": "x"}
        )
        cls.constant = OperationTree.objects.create(
            id=2, operation="constant", parameters={"value": 1}
        )
        cls.dataset.data_contents.add(cls.quantity)
        cls.client = APIClient()
        cls.client.force_authenticate(cls.user)

    # Test accessing a quantity with no operations performed
    def test_get_operation_tree_none(self):
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
                        "operation_tree": None,
                    }
                ],
            },
        )

    # Test accessing quantity with unary operation
    def test_get_operation_tree_unary(self):
        inv = OperationTree.objects.create(
            id=3, operation="reciprocal", parent_operation1=self.variable
        )
        self.quantity.operation_tree = inv
        self.quantity.save()
        request = self.client.get("/v1/data/set/1/")
        inv.delete()
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

    # Test accessing quantity with binary operation
    def test_get_operation_tree_binary(self):
        add = OperationTree.objects.create(
            id=3,
            operation="add",
            parent_operation1=self.variable,
            parent_operation2=self.constant,
        )
        self.quantity.operation_tree = add
        self.quantity.save()
        request = self.client.get("/v1/data/set/1/")
        add.delete()
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

    # Test accessing a quantity with exponent
    def test_get_operation_tree_pow(self):
        power = OperationTree.objects.create(
            id=3,
            operation="pow",
            parent_operation1=self.variable,
            parameters={"power": 2},
        )
        self.quantity.operation_tree = power
        self.quantity.save()
        request = self.client.get("/v1/data/set/1/")
        power.delete()
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

    # Test accessing a quantity with multiple operations
    def test_get_operation_tree_nested(self):
        multiply = OperationTree.objects.create(
            id=3,
            operation="mul",
            parent_operation1=self.constant,
            parent_operation2=self.variable,
        )
        neg = OperationTree.objects.create(
            id=4, operation="neg", parent_operation1=multiply
        )
        self.quantity.operation_tree = neg
        self.quantity.save()
        request = self.client.get("/v1/data/set/1/")
        multiply.delete()
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
                                            "parameters": {"value": 1},
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

    # Test accessing a transposed quantity
    def test_get_operation_tree_transpose(self):
        trans = OperationTree.objects.create(
            id=3,
            operation="transpose",
            parent_operation1=self.variable,
            parameters={"axes": (1, 0)},
        )
        self.quantity.operation_tree = trans
        self.quantity.save()
        request = self.client.get("/v1/data/set/1/")
        trans.delete()
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

    # Test accessing a quantity with tensordot

    @classmethod
    def tearDownClass(cls):
        cls.user.delete()
        cls.quantity.delete()
        cls.dataset.delete()
        cls.variable.delete()
        cls.constant.delete()
