
import numpy
from wykop_sdk_reloaded.v3.client import AuthClient, WykopApiClient

auth = AuthClient()


app_key = open('klucz_do_api.txt').read()
app_secret = open('secret_key.txt').read()

"""
autoryzując się w ten sposob masz tylko dostep do operacji odczytu.
Reszta wymaga WykopApiClient.authenticate_user()
""" 
auth.authenticate_app(app_key, app_secret)

api = WykopApiClient(auth)


w=api.entries_list_entries()
from wykop_sdk_reloaded.v3.types import *
l=api.w=api.links_list_links(type=LinkType.HOMEPAGE)

import json
def dump(obj, fname):
    with open(fname, 'w') as fp:
        json.dump(obj, fp)
dump(l, 'links.json')
znal_id = '7809697'
z = api.links_get_link(znal_id)
c=api.link_comments_list_comments(znal_id, LinkCommentSortType.NEWEST)
dump(c, 'comments_'+znal_id+".json")

comments = c['data']
for comm in comments:
    print(f"+{comm['votes']['up']:2d} -{comm['votes']['down']:2d} = {comm['votes']['up']-comm['votes']['down']} {comm['content'].replace(chr(10),'')[:150]}")
