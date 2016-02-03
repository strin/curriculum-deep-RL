import external.dropbox as dbx
import os

try:
    APP_KEY   = os.environ['DROPBOX_APP_KEY']
    APP_SECRET = os.environ['DROPBOX_APP_SECRET']
    TEST_ACCESS_TOKEN = os.environ['DROPBOX_TEST_ACCESS_TOKEN']
except:
    raise ImportError('cannot find dropbox api')

def _get_access_token():
    """
    Returns a url using which users can log in Dropbox and get access token
    for us.
    """
    flow = dbx.client.DropboxOAuth2FlowNoRedirect(APP_KEY, APP_SECRET)
    authorize_url = flow.start()
    return authorize_url

def put_file(path, stream, access_token=TEST_ACCESS_TOKEN):
    """
    Use the access_token to put the file in Dropbox path.
    """
    client = dbx.client.DropboxClient(access_token)
    response = client.put_file(path, stream)
    return response #(TODO) wrap the response.

def get_file(path, access_token=TEST_ACCESS_TOKEN):
    """
    Use the access_token to get file from path.
    Return a stream.
    """
    client = dbx.client.DropboxClient(access_token)
    stream, metadata = client.get_file_and_metadata(path)
    return stream

def shared_link(path, access_token=TEST_ACCESS_TOKEN):
    """
    get a shared link for a given path
    """
    client = dbx.client.DropboxClient(access_token)
    result = client.share(path, short_url=False)
    url = result[u'url'].replace('dl=0', 'dl=1') # direct link.
    return url

def account_info(access_token=TEST_ACCESS_TOKEN):
    client = dbx.client.DropboxClient(access_token)
    return client.account_info()


if __name__ == '__main__':
    print 'authorize_url', _get_access_token()

