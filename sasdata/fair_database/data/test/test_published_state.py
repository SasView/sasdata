from django.contrib.auth.models import User
from django.db.models import Max
from rest_framework import status
from rest_framework.test import APIClient, APITestCase

from data.models import PublishedState, Session


# TODO: account for non-placeholder doi
def doi_generator(id: int):
    return "http://127.0.0.1:8000/v1/data/session/" + str(id) + "/"


class TestPublishedState(APITestCase):
    """Test HTTP methods of PublishedStateView."""

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
        cls.unpublished_session = Session.objects.create(
            id=4, current_user=cls.user1, title="Publishable Session", is_public=True
        )
        cls.public_ps = PublishedState.objects.create(
            id=1,
            doi=doi_generator(1),
            published=True,
            session=cls.public_session,
        )
        cls.private_ps = PublishedState.objects.create(
            id=2,
            doi=doi_generator(2),
            published=False,
            session=cls.private_session,
        )
        cls.unowned_ps = PublishedState.objects.create(
            id=3,
            doi=doi_generator(3),
            published=True,
            session=cls.unowned_session,
        )
        cls.auth_client1 = APIClient()
        cls.auth_client2 = APIClient()
        cls.auth_client1.force_authenticate(cls.user1)
        cls.auth_client2.force_authenticate(cls.user2)

    # Test listing published states including those of owned private sessions
    def test_list_published_states_private(self):
        request = self.auth_client1.get("/v1/data/published/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data,
            {
                "published_state_ids": {
                    1: {
                        "title": "Public Session",
                        "published": True,
                        "doi": doi_generator(1),
                    },
                    2: {
                        "title": "Private Session",
                        "published": False,
                        "doi": doi_generator(2),
                    },
                    3: {
                        "title": "Unowned Session",
                        "published": True,
                        "doi": doi_generator(3),
                    },
                }
            },
        )

    # Test listing published states of public sessions
    def test_list_published_states_public(self):
        request = self.auth_client2.get("/v1/data/published/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data,
            {
                "published_state_ids": {
                    1: {
                        "title": "Public Session",
                        "published": True,
                        "doi": doi_generator(1),
                    },
                    3: {
                        "title": "Unowned Session",
                        "published": True,
                        "doi": doi_generator(3),
                    },
                }
            },
        )

    # Test listing published states including sessions with access granted
    def test_list_published_states_shared(self):
        self.private_session.users.add(self.user2)
        request = self.auth_client2.get("/v1/data/published/")
        self.private_session.users.remove(self.user2)
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data,
            {
                "published_state_ids": {
                    1: {
                        "title": "Public Session",
                        "published": True,
                        "doi": doi_generator(1),
                    },
                    2: {
                        "title": "Private Session",
                        "published": False,
                        "doi": doi_generator(2),
                    },
                    3: {
                        "title": "Unowned Session",
                        "published": True,
                        "doi": doi_generator(3),
                    },
                }
            },
        )

    # Test listing published states while unauthenticated
    def test_list_published_states_unauthenticated(self):
        request = self.client.get("/v1/data/published/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data,
            {
                "published_state_ids": {
                    1: {
                        "title": "Public Session",
                        "published": True,
                        "doi": doi_generator(1),
                    },
                    3: {
                        "title": "Unowned Session",
                        "published": True,
                        "doi": doi_generator(3),
                    },
                }
            },
        )

    # Test listing a user's own published states
    def test_list_user_published_states_private(self):
        request = self.auth_client1.get(
            "/v1/data/published/", data={"username": "testUser1"}
        )
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data,
            {
                "published_state_ids": {
                    1: {
                        "title": "Public Session",
                        "published": True,
                        "doi": doi_generator(1),
                    },
                    2: {
                        "title": "Private Session",
                        "published": False,
                        "doi": doi_generator(2),
                    },
                }
            },
        )

    # Test listing another user's published states
    def test_list_user_published_states_public(self):
        request = self.auth_client2.get(
            "/v1/data/published/", data={"username": "testUser1"}
        )
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data,
            {
                "published_state_ids": {
                    1: {
                        "title": "Public Session",
                        "published": True,
                        "doi": doi_generator(1),
                    }
                }
            },
        )

    # Test listing another user's published states with access granted
    def test_list_user_published_states_shared(self):
        self.private_session.users.add(self.user2)
        request = self.auth_client2.get(
            "/v1/data/published/", data={"username": "testUser1"}
        )
        self.private_session.users.remove(self.user2)
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data,
            {
                "published_state_ids": {
                    1: {
                        "title": "Public Session",
                        "published": True,
                        "doi": doi_generator(1),
                    },
                    2: {
                        "title": "Private Session",
                        "published": False,
                        "doi": doi_generator(2),
                    },
                }
            },
        )

    # Test listing a user's published states while unauthenticated
    def test_list_user_published_states_unauthenticated(self):
        request = self.client.get("/v1/data/published/", data={"username": "testUser1"})
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data,
            {
                "published_state_ids": {
                    1: {
                        "title": "Public Session",
                        "published": True,
                        "doi": doi_generator(1),
                    }
                }
            },
        )

    # Test creating a published state for a private session
    def test_published_state_created_private(self):
        self.unpublished_session.is_public = False
        self.unpublished_session.save()
        published_state = {"published": True, "session": 4}
        request = self.auth_client1.post("/v1/data/published/", data=published_state)
        max_id = PublishedState.objects.aggregate(Max("id"))["id__max"]
        new_ps = PublishedState.objects.get(id=max_id)
        self.publishable_session = Session.objects.get(id=4)
        self.assertEqual(request.status_code, status.HTTP_201_CREATED)
        self.assertEqual(
            request.data,
            {
                "published_state_id": max_id,
                "session_id": 4,
                "title": "Publishable Session",
                "doi": doi_generator(4),
                "published": True,
                "current_user": "testUser1",
                "is_public": False,
            },
        )
        self.assertEqual(self.publishable_session.published_state, new_ps)
        self.assertEqual(new_ps.session, self.publishable_session)
        new_ps.delete()
        self.unpublished_session.is_public = True
        self.unpublished_session.save()

    # Test creating a published state for a public session
    def test_published_state_created_public(self):
        published_state = {"published": False, "session": 4}
        request = self.auth_client1.post("/v1/data/published/", data=published_state)
        max_id = PublishedState.objects.aggregate(Max("id"))["id__max"]
        new_ps = PublishedState.objects.get(id=max_id)
        self.publishable_session = Session.objects.get(id=4)
        self.assertEqual(request.status_code, status.HTTP_201_CREATED)
        self.assertEqual(
            request.data,
            {
                "published_state_id": max_id,
                "session_id": 4,
                "title": "Publishable Session",
                "doi": doi_generator(4),
                "published": False,
                "current_user": "testUser1",
                "is_public": True,
            },
        )
        self.assertEqual(self.publishable_session.published_state, new_ps)
        self.assertEqual(new_ps.session, self.publishable_session)
        new_ps.delete()

    # Test that you can't create a published state for an unowned session
    def test_published_state_created_unowned(self):
        self.unpublished_session.current_user = None
        self.unpublished_session.save()
        published_state = {"published": True, "session": 4}
        request = self.auth_client1.post("/v1/data/published/", data=published_state)
        self.assertEqual(request.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(len(PublishedState.objects.all()), 3)
        self.unpublished_session.current_user = self.user1
        self.unpublished_session.save()

    # Test that an unauthenticated user cannot create a published state
    def test_published_state_created_unauthenticated(self):
        published_state = {"published": True, "session": 4}
        request = self.client.post("/v1/data/published/", data=published_state)
        self.assertEqual(request.status_code, status.HTTP_401_UNAUTHORIZED)
        self.assertEqual(len(PublishedState.objects.all()), 3)

    # Test that a user cannot create a published state for a session they don't own
    def test_published_state_created_unauthorized(self):
        published_state = {"published": True, "session": 4}
        request = self.auth_client2.post("/v1/data/published/", data=published_state)
        self.assertEqual(request.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(len(PublishedState.objects.all()), 3)

    # Test that only one published state can be created per session
    def test_no_duplicate_published_states(self):
        published_state = {"published": True, "session": 1}
        request = self.auth_client1.post("/v1/data/published/", data=published_state)
        self.assertEqual(request.status_code, status.HTTP_400_BAD_REQUEST)

    @classmethod
    def tearDownClass(cls):
        cls.public_session.delete()
        cls.private_session.delete()
        cls.unowned_session.delete()
        cls.user1.delete()
        cls.user2.delete()


class TestSinglePublishedState(APITestCase):
    """Test HTTP methods of SinglePublishedStateView."""

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
        cls.public_ps = PublishedState.objects.create(
            id=1,
            doi=doi_generator(1),
            published=True,
            session=cls.public_session,
        )
        cls.private_ps = PublishedState.objects.create(
            id=2,
            doi=doi_generator(2),
            published=False,
            session=cls.private_session,
        )
        cls.unowned_ps = PublishedState.objects.create(
            id=3,
            doi=doi_generator(3),
            published=True,
            session=cls.unowned_session,
        )
        cls.auth_client1 = APIClient()
        cls.auth_client2 = APIClient()
        cls.auth_client1.force_authenticate(cls.user1)
        cls.auth_client2.force_authenticate(cls.user2)

    # Test viewing a published state of a public session
    def test_get_public_published_state(self):
        request1 = self.auth_client2.get("/v1/data/published/1/")
        request2 = self.client.get("/v1/data/published/1/")
        self.assertEqual(request1.status_code, status.HTTP_200_OK)
        self.assertEqual(request2.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request1.data,
            {
                "id": 1,
                "doi": doi_generator(1),
                "published": True,
                "session": 1,
                "title": "Public Session",
                "current_user": "testUser1",
                "is_public": True,
            },
        )
        self.assertEqual(request1.data, request2.data)

    # Test viewing a published state of a private session
    def test_get_private_published_state(self):
        request = self.auth_client1.get("/v1/data/published/2/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data,
            {
                "id": 2,
                "doi": doi_generator(2),
                "published": False,
                "session": 2,
                "title": "Private Session",
                "current_user": "testUser1",
                "is_public": False,
            },
        )

    # Test viewing a published state of an unowned session
    def test_get_unowned_published_state(self):
        request = self.auth_client1.get("/v1/data/published/3/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data,
            {
                "id": 3,
                "doi": doi_generator(3),
                "published": True,
                "session": 3,
                "title": "Unowned Session",
                "current_user": "",
                "is_public": True,
            },
        )

    # Test viewing a published state of a session with access granted
    def test_get_shared_published_state(self):
        self.private_session.users.add(self.user2)
        request = self.auth_client2.get("/v1/data/published/2/")
        self.private_session.users.remove(self.user2)
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data,
            {
                "id": 2,
                "doi": doi_generator(2),
                "published": False,
                "session": 2,
                "title": "Private Session",
                "current_user": "testUser1",
                "is_public": False,
            },
        )

    # Test a user can't view a published state of a private session they don't own
    def test_get_private_published_state_unauthorized(self):
        request1 = self.client.get("/v1/data/published/2/")
        request2 = self.auth_client2.get("/v1/data/published/2/")
        self.assertEqual(request1.status_code, status.HTTP_401_UNAUTHORIZED)
        self.assertEqual(request2.status_code, status.HTTP_403_FORBIDDEN)

    # Test updating a published state of a public session
    def test_update_public_published_state(self):
        request = self.auth_client1.put(
            "/v1/data/published/1/", data={"published": False}
        )
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data,
            {
                "published_state_id": 1,
                "session_id": 1,
                "title": "Public Session",
                "published": False,
                "is_public": True,
            },
        )
        self.assertFalse(PublishedState.objects.get(id=1).published)
        self.public_ps.save()

    # Test updating a published state of a private session
    def test_update_private_published_state(self):
        request = self.auth_client1.put(
            "/v1/data/published/2/", data={"published": True}
        )
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(
            request.data,
            {
                "published_state_id": 2,
                "session_id": 2,
                "title": "Private Session",
                "published": True,
                "is_public": False,
            },
        )
        self.assertTrue(PublishedState.objects.get(id=2).published)
        self.private_ps.save()

    # Test a user can't update the published state of an unowned session
    def test_update_unowned_published_state(self):
        request1 = self.auth_client1.put(
            "/v1/data/published/3/", data={"published": False}
        )
        request2 = self.client.put("/v1/data/published/3/", data={"published": False})
        self.assertEqual(request1.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(request2.status_code, status.HTTP_401_UNAUTHORIZED)
        self.assertTrue(PublishedState.objects.get(id=3).published)

    # Test a user can't update a public published state unauthorized
    def test_update_public_published_state_unauthorized(self):
        request1 = self.auth_client2.put(
            "/v1/data/published/1/", data={"published": False}
        )
        self.public_session.users.add(self.user2)
        request2 = self.auth_client2.put(
            "/v1/data/published/1/", data={"published": False}
        )
        self.public_session.users.remove(self.user2)
        request3 = self.client.put("/v1/data/published/1/", data={"published": False})
        self.assertEqual(request1.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(request2.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(request3.status_code, status.HTTP_401_UNAUTHORIZED)
        self.assertTrue(PublishedState.objects.get(id=1).published)

    # Test a user can't update a private published state unauthorized
    def test_update_private_published_state_unauthorized(self):
        request1 = self.auth_client2.put(
            "/v1/data/published/2/", data={"published": True}
        )
        self.public_session.users.add(self.user2)
        request2 = self.auth_client2.put(
            "/v1/data/published/2/", data={"published": True}
        )
        self.public_session.users.remove(self.user2)
        request3 = self.client.put("/v1/data/published/2/", data={"published": True})
        self.assertEqual(request1.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(request2.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(request3.status_code, status.HTTP_401_UNAUTHORIZED)
        self.assertFalse(PublishedState.objects.get(id=2).published)

    # Test deleting a published state of a private session
    def test_delete_private_published_state(self):
        request = self.auth_client1.delete("/v1/data/published/2/")
        self.assertEqual(request.status_code, status.HTTP_200_OK)
        self.assertEqual(len(PublishedState.objects.all()), 2)
        self.assertEqual(len(Session.objects.all()), 3)
        self.assertRaises(PublishedState.DoesNotExist, PublishedState.objects.get, id=2)
        self.private_ps = PublishedState.objects.create(
            id=2,
            doi=doi_generator(2),
            published=False,
            session=self.private_session,
        )

    # Test a user can't delete a private published state unauthorized
    def test_delete_private_published_state_unauthorized(self):
        request1 = self.auth_client2.delete("/v1/data/published/2/")
        self.private_session.users.add(self.user2)
        request2 = self.auth_client2.delete("/v1/data/published/2/")
        self.private_session.users.remove(self.user2)
        request3 = self.client.delete("/v1/data/published/2/")
        self.assertEqual(request1.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(request2.status_code, status.HTTP_403_FORBIDDEN)
        self.assertEqual(request3.status_code, status.HTTP_401_UNAUTHORIZED)

    # Test a user can't delete a published state of a public
    def test_cant_delete_public_published_state(self):
        request = self.auth_client1.delete("/v1/data/published/1/")
        self.assertEqual(request.status_code, status.HTTP_403_FORBIDDEN)

    # Test a user can't delete an unowned published state
    def test_delete_unowned_published_state(self):
        request = self.auth_client1.delete("/v1/data/published/3/")
        self.assertEqual(request.status_code, status.HTTP_403_FORBIDDEN)

    @classmethod
    def tearDownClass(cls):
        cls.public_session.delete()
        cls.private_session.delete()
        cls.unowned_session.delete()
        cls.user1.delete()
        cls.user2.delete()
