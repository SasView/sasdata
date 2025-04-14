from django.contrib.auth.models import User
from rest_framework.test import APIClient, APITestCase
from rest_framework import status


from data.models import Session


class TestSession(APITestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user1 = User.objects.create_user(
            id=1, username="testUser1", password="secret"
        )
        cls.user2 = User.objects.create_user(
            id=2, username="testUser2", password="secret"
        )
        cls.public_session = Session.objects.create(
            id=1, current_user=cls.user1, title="Public Session", is_public=True
        )
        cls.private_session = Session.objects.create(
            id=2, current_user=cls.user1, title="Private Session", is_public=False
        )
        cls.unowned_session = Session.objects.create(
            id=3, title="Unowned Session", is_public=True
        )
        cls.auth_client1 = APIClient()
        cls.auth_client2 = APIClient()
        cls.auth_client1.force_authenticate(cls.user1)
        cls.auth_client2.force_authenticate(cls.user2)

    # Test listing sessions
    def test_list_private(self):
        request = self.auth_client1.get("/v1/data/session/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data,
            {
                "session_ids": {
                    1: "Public Session",
                    2: "Private Session",
                    3: "Unowned Session",
                }
            },
        )

    # Test listing public sessions
    def test_list_public(self):
        request = self.auth_client2.get("/v1/data/session/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data, {"session_ids": {1: "Public Session", 3: "Unowned Session"}}
        )

    # Test listing sessions while unauthenticated
    def test_list_unauthenticated(self):
        request = self.client.get("/v1/data/session/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data, {"session_ids": {1: "Public Session", 3: "Unowned Session"}}
        )

    # Test listing a session with access granted
    def test_list_granted_access(self):
        self.private_session.users.add(self.user2)
        request = self.auth_client2.get("/v1/data/session/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data,
            {
                "session_ids": {
                    1: "Public Session",
                    2: "Private Session",
                    3: "Unowned Session",
                }
            },
        )
        self.private_session.users.remove(self.user2)

    # Test listing by username
    def test_list_username(self):
        request = self.auth_client1.get(
            "/v1/data/session/", data={"username": "testUser1"}
        )
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data, {"session_ids": {1: "Public Session", 2: "Private Session"}}
        )

    # Test listing by another user's username
    def test_list_other_username(self):
        request = self.auth_client2.get(
            "/v1/data/session/", data={"username": "testUser1"}
        )
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(request.data, {"session_ids": {1: "Public Session"}})

    # Test creating a session - public, private, unauthenticated
    # Datasets have same access as session

    # Test post fails with dataset validation issue

    @classmethod
    def tearDownClass(cls):
        cls.public_session.delete()
        cls.private_session.delete()
        cls.unowned_session.delete()
        cls.user1.delete()
        cls.user2.delete()


class TestSingleSession(APITestCase):
    @classmethod
    def setUpTestData(cls):
        pass

    # Test accessing session
    # public, private, with/without access

    # Test updating session
    # public, private, owner, not owner

    # Test deleting session
    # Test delete cascades

    @classmethod
    def tearDownClass(cls):
        pass


class TestSessionAccessManagement(APITestCase):
    @classmethod
    def setUpTestData(cls):
        pass

    # Test listing access

    # Test listing access not as the owner

    # Test granting access

    # Test revoking access

    # Test can't revoke own access

    # Test only owner can change access

    @classmethod
    def tearDownClass(cls):
        pass
