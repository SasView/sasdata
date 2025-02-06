import os

from django.contrib.auth.models import User
from rest_framework.test import APIClient, APITestCase

from data.models import Data

def find(filename):
    return os.path.join(os.path.dirname(__file__), "../../example_data/1d_data", filename)

class DataListPermissionsTests(APITestCase):
    ''' Test permissions of data views using user_app for authentication. '''

    def setUp(self):
        self.user = User.objects.create_user(username="testUser", password="secret", id=1)
        self.user2 = User.objects.create_user(username="testUser2", password="secret", id=2)
        public_test_data = Data.objects.create(id=1, file_name="cyl_400_40.txt",
                                               is_public=True)
        public_test_data.file.save("cyl_400_40.txt", open(find("cyl_400_40.txt"), 'rb'))
        private_test_data = Data.objects.create(id=2, current_user=self.user,
                                                file_name="cyl_400_20.txt", is_public=False)
        private_test_data.file.save("cyl_400_20.txt", open(find("cyl_400_20.txt"), 'rb'))

    # Authenticated user can view list of data
    
    # Unauthenticated user can view list of public data

    # Authenticated user cannot view other users' private data on list

    # Authenticated user can load public data

    # Authenticated user can load own private data

    # Authenticated user cannot load others' private data

    # Unauthenticated user can load public data

    # Unauthenticated user cannot load others' private data

    # Authenticated user can upload data

    # ***Unauthenticated user can upload public data

    # Unauthenticated user cannot upload private data

    # Authenticated user can update own public data

    # Authenticated user can update own private data

    # Authenticated user cannot update unowned public data

    # Unauthenticated user cannot update data

    # Anyone can download public data

    # Authenticated user can download own data

    # Authenticated user cannot download others' data

    # Unauthenticated user cannot download private data
