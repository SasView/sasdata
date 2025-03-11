from django.contrib.auth.models import User
from rest_framework.test import APIClient, APITestCase

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
        cls.user1 = User.objects.create_user(username="testUser1", password="secret")
        cls.user2 = User.objects.create_user(username="testUser2", password="secret")
        cls.public_dataset = DataSet.objects.create(
            id=1,
            current_user=cls.user1,
            is_public=True,
            name="Dataset 1",
            metadata=None,
        )
        cls.private_dataset = DataSet.objects.create(
            id=2, current_user=cls.user1, name="Dataset 1", metadata=None
        )
        cls.unowned_dataset = DataSet.objects.create(
            id=3, is_public=True, name="Dataset 3", metadata=None
        )
        cls.auth_client1 = APIClient()
        cls.auth_client2 = APIClient()
        cls.auth_client1.force_authenticate(cls.user1)
        cls.auth_client2.force_authenticate(cls.user2)

    @classmethod
    def tearDownClass(cls):
        cls.public_dataset.delete()
        cls.private_dataset.delete()
        cls.unowned_dataset.delete()
        cls.user1.delete()
        cls.user2.delete()


# TODO: decide whether to just combine this w/ above
class TestSingleDataSet(APITestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user1 = User.objects.create_user(username="testUser1", password="secret")
        cls.user2 = User.objects.create_user(username="testUser2", password="secret")
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
