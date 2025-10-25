from wykop_sdk_reloaded.v3.client import AuthClient, WykopApiClient
from wykop_sdk_reloaded.v3.types import LinkType, LinkCommentSortType
import json

auth = AuthClient()


app_key = open("klucz_do_api.txt").read()
app_secret = open("secret_key.txt").read()

"""
autoryzując się w ten sposob masz tylko dostep do operacji odczytu.
Reszta wymaga WykopApiClient.authenticate_user()
"""
auth.authenticate_app(app_key, app_secret)

api = WykopApiClient(auth)


w = api.entries_list_entries()

l = api.w = api.links_list_links(type=LinkType.HOMEPAGE, page=2)


def dump(obj, fname):
    with open(fname, "w") as fp:
        json.dump(obj, fp)


dump(l, "links.json")

znal_id_1 = "6893651"
z = api.links_get_link(znal_id_1)
dump(z, "link_" + znal_id_1 + ".json")

c = api.link_comments_list_comments(znal_id_1, LinkCommentSortType.NEWEST)
dump(c, "comments_" + znal_id_1 + ".json")


comments = c["data"]
for comm in comments:
    print(
        f"+{comm['votes']['up']:2d} -{comm['votes']['down']:2d} = {comm['votes']['up']-comm['votes']['down']} {comm['content'].replace(chr(10),'')[:150]}"
    )
