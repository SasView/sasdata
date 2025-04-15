from django.contrib.auth.models import User
from django.db.models import Max
from rest_framework.test import APIClient, APITestCase
from rest_framework import status


from data.models import DataSet, Session


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

    # Test creating a public session
    def test_session_created(self):
        session = {
            "title": "New session",
            "datasets": [
                {
                    "name": "New dataset",
                    "metadata": {
                        "title": "New metadata",
                        "run": 0,
                        "description": "test",
                        "instrument": {},
                        "process": {},
                        "sample": {},
                    },
                    "data_contents": [],
                }
            ],
            "is_public": True,
        }
        request = self.auth_client1.post(
            "/v1/data/session/", data=session, format="json"
        )
        max_id = Session.objects.aggregate(Max("id"))["id__max"]
        new_session = Session.objects.get(id=max_id)
        new_dataset = new_session.datasets.get()
        new_metadata = new_dataset.metadata
        self.assertEqual(request.status_code, status.HTTP_201_CREATED)
        self.assertEqual(
            request.data,
            {
                "session_id": max_id,
                "title": "New session",
                "authenticated": True,
                "current_user": "testUser1",
                "is_public": True,
            },
        )
        self.assertEqual(new_session.title, "New session")
        self.assertEqual(new_dataset.name, "New dataset")
        self.assertEqual(new_metadata.title, "New metadata")
        self.assertEqual(new_session.current_user, self.user1)
        self.assertEqual(new_dataset.current_user, self.user1)
        self.assertTrue(all([new_session.is_public, new_dataset.is_public]))
        new_session.delete()

    # Test creating a private session
    def test_session_created_private(self):
        session = {
            "title": "New session",
            "datasets": [
                {
                    "name": "New dataset",
                    "metadata": {
                        "title": "New metadata",
                        "run": 0,
                        "description": "test",
                        "instrument": {},
                        "process": {},
                        "sample": {},
                    },
                    "data_contents": [],
                }
            ],
            "is_public": False,
        }
        request = self.auth_client1.post(
            "/v1/data/session/", data=session, format="json"
        )
        max_id = Session.objects.aggregate(Max("id"))["id__max"]
        new_session = Session.objects.get(id=max_id)
        new_dataset = new_session.datasets.get()
        self.assertEqual(request.status_code, status.HTTP_201_CREATED)
        self.assertEqual(
            request.data,
            {
                "session_id": max_id,
                "title": "New session",
                "authenticated": True,
                "current_user": "testUser1",
                "is_public": False,
            },
        )
        self.assertEqual(new_session.current_user, self.user1)
        self.assertEqual(new_dataset.current_user, self.user1)
        self.assertTrue(all([(not new_session.is_public), (not new_dataset.is_public)]))
        new_session.delete()

    # Test creating a session while unauthenticated
    def test_session_created_unauthenticated(self):
        session = {
            "title": "New session",
            "datasets": [
                {
                    "name": "New dataset",
                    "metadata": {
                        "title": "New metadata",
                        "run": 0,
                        "description": "test",
                        "instrument": {},
                        "process": {},
                        "sample": {},
                    },
                    "data_contents": [],
                }
            ],
            "is_public": True,
        }
        request = self.client.post("/v1/data/session/", data=session, format="json")
        max_id = Session.objects.aggregate(Max("id"))["id__max"]
        new_session = Session.objects.get(id=max_id)
        new_dataset = new_session.datasets.get()
        self.assertEqual(request.status_code, status.HTTP_201_CREATED)
        self.assertEqual(
            request.data,
            {
                "session_id": max_id,
                "title": "New session",
                "authenticated": False,
                "current_user": "",
                "is_public": True,
            },
        )
        self.assertIsNone(new_session.current_user)
        self.assertIsNone(new_dataset.current_user)
        self.assertTrue(all([new_session.is_public, new_dataset.is_public]))
        new_session.delete()

    # Test that a private session must have an owner
    def test_no_private_unowned_session(self):
        session = {"title": "New session", "datasets": [], "is_public": False}
        request = self.client.post("/v1/data/session/", data=session, format="json")
        self.assertEqual(request.status_code, status.HTTP_400_BAD_REQUEST)

    # Test post fails with dataset validation issue
    def test_no_session_invalid_dataset(self):
        session = {
            "title": "New session",
            "datasets": [
                {
                    "metadata": {
                        "title": "New metadata",
                        "run": 0,
                        "description": "test",
                        "instrument": {},
                        "process": {},
                        "sample": {},
                    },
                    "data_contents": [],
                }
            ],
            "is_public": True,
        }
        request = self.auth_client1.post(
            "/v1/data/session/", data=session, format="json"
        )
        self.assertEqual(request.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(len(Session.objects.all()), 3)
        self.assertEqual(len(DataSet.objects.all()), 0)

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
