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
        self.assertFalse(any([new_session.is_public, new_dataset.is_public]))
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
        cls.public_dataset = DataSet.objects.create(
            id=1,
            current_user=cls.user1,
            is_public=True,
            name="Public Dataset",
            session=cls.public_session,
        )
        cls.private_dataset = DataSet.objects.create(
            id=2,
            current_user=cls.user1,
            name="Private Dataset",
            session=cls.private_session,
        )
        cls.unowned_dataset = DataSet.objects.create(
            id=3, is_public=True, name="Unowned Dataset", session=cls.unowned_session
        )
        cls.auth_client1 = APIClient()
        cls.auth_client2 = APIClient()
        cls.auth_client1.force_authenticate(cls.user1)
        cls.auth_client2.force_authenticate(cls.user2)

    # Test loading another user's public session
    def test_get_public_session(self):
        request = self.auth_client2.get("/v1/data/session/1/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data,
            {
                "id": 1,
                "current_user": 1,
                "users": [],
                "is_public": True,
                "title": "Public Session",
                "published_state": None,
                "datasets": [
                    {
                        "id": 1,
                        "current_user": 1,
                        "users": [],
                        "is_public": True,
                        "name": "Public Dataset",
                        "files": [],
                        "metadata": None,
                        "data_contents": [],
                    }
                ],
            },
        )

    # Test loading a private session as the owner
    def test_get_private_session(self):
        request = self.auth_client1.get("/v1/data/session/2/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data,
            {
                "id": 2,
                "current_user": 1,
                "users": [],
                "is_public": False,
                "title": "Private Session",
                "published_state": None,
                "datasets": [
                    {
                        "id": 2,
                        "current_user": 1,
                        "users": [],
                        "is_public": False,
                        "name": "Private Dataset",
                        "files": [],
                        "metadata": None,
                        "data_contents": [],
                    }
                ],
            },
        )

    # Test loading a private session as a user with granted access
    def test_get_private_session_access_granted(self):
        self.private_session.users.add(self.user2)
        request = self.auth_client2.get("/v1/data/session/2/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.private_session.users.remove(self.user2)

    # Test loading an unowned session
    def test_get_unowned_session(self):
        request = self.auth_client1.get("/v1/data/session/3/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data,
            {
                "id": 3,
                "current_user": None,
                "users": [],
                "is_public": True,
                "title": "Unowned Session",
                "published_state": None,
                "datasets": [
                    {
                        "id": 3,
                        "current_user": None,
                        "users": [],
                        "is_public": True,
                        "name": "Unowned Dataset",
                        "files": [],
                        "metadata": None,
                        "data_contents": [],
                    }
                ],
            },
        )

    # Test loading another user's private session
    def test_get_private_session_unauthorized(self):
        request1 = self.auth_client2.get("/v1/data/session/2/")
        request2 = self.client.get("/v1/data/session/2/")
        self.assertEqual(request1.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(request2.status_code, status.HTTP_401_UNAUTHORIZED)

    # Test updating session
    # public, private, owner, not owner
    def test_update_public_session(self):
        request = self.auth_client1.put(
            "/v1/data/session/1/", data={"is_public": False}
        )
        session = Session.objects.get(id=1)
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data,
            {"session_id": 1, "title": "Public Session", "is_public": False},
        )
        self.assertFalse(session.is_public)
        session.is_public = False
        session.save()

    def test_update_public_session_unauthorized(self):
        request1 = self.auth_client2.put(
            "/v1/data/session/1/", data={"is_public": False}
        )
        request2 = self.client.put("/v1/data/session/1/", data={"is_public": False})
        session = Session.objects.get(id=1)
        session.users.add(self.user2)
        request3 = self.auth_client2.put(
            "/v1/data/session/1/", data={"is_public": False}
        )
        session.users.remove(self.user2)
        self.assertEqual(request1.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(request2.status_code, status.HTTP_401_UNAUTHORIZED)
        self.assertEqual(request3.status_code, status.HTTP_403_FORBIDDEN)
        self.assertTrue(Session.objects.get(id=1).is_public)

    def test_update_private_session(self):
        request1 = self.auth_client1.put(
            "/v1/data/session/2/", data={"is_public": True}
        )
        session = Session.objects.get(id=2)
        self.assertEqual(request1.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request1.data,
            {"session_id": 2, "title": "Private Session", "is_public": True},
        )
        self.assertTrue(session.is_public)
        session.is_public = False
        session.save()

    def test_update_private_session_unauthorized(self):
        request1 = self.auth_client2.put(
            "/v1/data/session/2/", data={"is_public": True}
        )
        request2 = self.client.put("/v1/data/session/2/", data={"is_public": True})
        session = Session.objects.get(id=2)
        session.users.add(self.user2)
        request3 = self.auth_client2.put(
            "/v1/data/session/2/", data={"is_public": True}
        )
        session.users.remove(self.user2)
        self.assertEqual(request1.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(request2.status_code, status.HTTP_401_UNAUTHORIZED)
        self.assertEqual(request3.status_code, status.HTTP_403_FORBIDDEN)
        self.assertFalse(Session.objects.get(id=2).is_public)

    def test_update_unowned_session(self):
        request = self.auth_client1.put(
            "/v1/data/session/3/", data={"is_public": False}
        )
        self.assertEqual(request.status_code, status.HTTP_403_FORBIDDEN)
        self.assertTrue(Session.objects.get(id=3).is_public)

    # Test deleting session
    # Test delete cascades
    def test_delete_private_session(self):
        request = self.auth_client1.delete("/v1/data/session/2/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertRaises(Session.DoesNotExist, Session.objects.get, id=2)
        self.assertRaises(DataSet.DoesNotExist, DataSet.objects.get, id=2)
        self.private_session = Session.objects.create(
            id=2, current_user=self.user1, title="Private Session", is_public=False
        )
        self.private_dataset = DataSet.objects.create(
            id=2,
            current_user=self.user1,
            name="Private Dataset",
            session=self.private_session,
        )

    def test_delete_private_session_unauthorized(self):
        request1 = self.auth_client2.delete("/v1/data/session/2/")
        request2 = self.client.delete("/v1/data/session/2/")
        self.private_session.users.add(self.user2)
        request3 = self.auth_client2.delete("/v1/data/session/2/")
        self.private_session.users.remove(self.user2)
        self.assertEqual(request1.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(request2.status_code, status.HTTP_401_UNAUTHORIZED)
        self.assertEqual(request3.status_code, status.HTTP_403_FORBIDDEN)

    def test_delete_public_session(self):
        request = self.auth_client1.delete("/v1/data/session/1/")
        self.assertEqual(request.status_code, status.HTTP_403_FORBIDDEN)

    def test_delete_unowned_session(self):
        request = self.auth_client1.delete("/v1/data/session/3/")
        self.assertEqual(request.status_code, status.HTTP_403_FORBIDDEN)

    @classmethod
    def tearDownClass(cls):
        cls.public_session.delete()
        cls.private_session.delete()
        cls.unowned_session.delete()
        cls.user1.delete()
        cls.user2.delete()


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
