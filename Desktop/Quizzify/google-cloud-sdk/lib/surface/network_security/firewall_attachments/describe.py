# -*- coding: utf-8 -*- #
# Copyright 2023 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Describe attachment command."""

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

from googlecloudsdk.api_lib.network_security.firewall_attachments import attachment_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.network_security import attachment_flags

DETAILED_HELP = {
    'DESCRIPTION': """
          Describe a firewall attachment.

          For more examples, refer to the EXAMPLES section below.

        """,
    'EXAMPLES': """
            To get a description of a firewall attachment called `my-attachment`, in zone
            `us-central1-a` and project my-proj, run:

            $ {command} my-attachment --zone=us-central1-a --project=my-proj

            $ {command} projects/my-proj/locations/us-central1-a/firewallAttachments/my-attachment

        """,
}


@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class Describe(base.DescribeCommand):
  """Describe a Firewall attachment."""

  @classmethod
  def Args(cls, parser):
    attachment_flags.AddAttachmentResource(cls.ReleaseTrack(), parser)

  def Run(self, args):
    client = attachment_api.Client(self.ReleaseTrack())

    attachment = args.CONCEPTS.firewall_attachment.Parse()

    return client.DescribeAttachment(attachment.RelativeName())


Describe.detailed_help = DETAILED_HELP
