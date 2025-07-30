from django.utils.decorators import method_decorator
from drf_yasg.utils import swagger_auto_schema
from rest_framework import generics
from rest_framework.permissions import IsAuthenticated

from .models import SessionTimeoutPolicy
from .serializers import SessionTimeoutPolicySerializer


@method_decorator(
    name='get',
    decorator=swagger_auto_schema(
        tags=['Session Policy'],
        operation_summary='Retrieve Session Policy',
        operation_description='Retrieve session timeout policy for the currently active organization.',
    ),
)
@method_decorator(
    name='patch',
    decorator=swagger_auto_schema(
        tags=['Session Policy'],
        operation_summary='Update Session Policy',
        operation_description='Update session timeout policy for the currently active organization.',
    ),
)
class SessionTimeoutPolicyView(generics.RetrieveUpdateAPIView):
    """
    API endpoint for retrieving and updating organization's session timeout policy
    """

    serializer_class = SessionTimeoutPolicySerializer
    permission_classes = [IsAuthenticated]
    http_method_names = ['get', 'patch']  # Explicitly specify allowed methods

    def get_object(self):
        # Get the organization from the request
        org = self.request.user.active_organization
        # Get or create the session policy for the organization
        policy, _ = SessionTimeoutPolicy.objects.get_or_create(organization=org)
        return policy
