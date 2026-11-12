# Publishing to YouTube, TikTok, Instagram and Facebook

Clip Studio uploads through each platform's official API. To do that, you register **your own developer app** on each platform once, add its keys under **API credentials** (top right), add your account under **Accounts**, and click **Connect**.

## API credentials and accounts
The header has two menus:
- **API credentials**: the keys of each developer app you registered (client ID and secret). The steps and the redirect URI to register are shown when you add them. The redirect URI is the same for every set of credentials on a platform.
- **Accounts**: the YouTube channels, TikTok accounts, Instagram accounts and Facebook Pages you post to. Add as many as you need. Each one picks which API credentials it signs in with, so one app can serve many accounts. If a platform has no credentials yet, adding an account takes you to set them up first.

Notes:
- **Login email / username:** each account can store the email or username it signs in with (never a password; the platforms only allow signing in on their own page). For YouTube, sign-in goes straight to that Google account. For TikTok and Instagram it's a reminder of which login to use in the popup.
- **Picking a YouTube channel:** a Google login can own several channels (brand accounts), and an upload goes to the channel you pick in Google's chooser when you click **Connect**. Google always shows the chooser, and each channel appears there as its own entry. To pin an account to one channel, enter its **YouTube channel** (`@handle`, `UC…` channel ID or youtube.com link) in the account settings. Clip Studio then won't connect a different channel, and before every upload it checks that the login still uploads to that channel. Each channel you post to needs its own account in Clip Studio; in the publish panel, tick the ones the clip should go to.
- **Signing in to a different account:** YouTube always shows the Google account chooser (a login email just preselects it). TikTok and Facebook may sign you straight into the account you're already logged into; sign out on their website first, then click **Connect**.
- **Instagram:** one Facebook login can manage several Instagram accounts. Put the Instagram username on each account so Clip Studio knows which one it is. Left empty, it takes the first one that no other account here uses.
- **Facebook:** one Facebook login can manage several Pages. Put the Page name (or ID) on each account. Left empty, it takes the first Page that no other account here uses.
- **Changing an app's keys** signs out the accounts that use it. Renaming doesn't.
- Credentials can't be removed while an account still uses them.
- In the **Publish** panel, tick each account to post to. The text you write for a platform is used for all of its ticked accounts.

Older setups are converted automatically the first time the server reads `data/accounts.json`.

## How it works
1. **Write titles & descriptions with AI.** The local vision model (qwen2.5vl via Ollama) writes per-platform text from 3 frames, the transcript, your on-screen text and the source file name:
   - YouTube: title + description + tags
   - TikTok: caption + hashtags
   - Instagram: caption + hashtags
   - Facebook: caption + hashtags

   **Read it before posting.** It can misread scenes, for example calling a water track "icy".
2. Tick the platforms you want, choose visibility, then click **Publish to selected**.
3. The page renders the clip and the server converts it to MP4 (H.264/AAC, 30 fps). The server then uploads it to each platform in parallel and shows a link when done.

## YouTube Shorts
1. In [Google Cloud Console](https://console.cloud.google.com/), create a project and enable **YouTube Data API v3**.
2. Under **OAuth consent screen**, choose **External** and add your Google account under **Test users**.
3. Under **Credentials → Create credentials → OAuth client ID**, choose **Web application**. Add this to **Authorized redirect URIs**:
   `http://localhost:8000/api/auth/youtube/callback`
4. Paste the client ID and secret into the panel and click **Connect**.

**Limits**
- Until Google audits your API project, every upload is locked to **Private**. After uploading, you can change it to Public on youtube.com.
- The default quota (10,000 units a day) covers about 6 uploads a day.
- Vertical videos up to 3 minutes become Shorts.

## TikTok
1. On [TikTok for Developers](https://developers.tiktok.com/apps), create an app.
2. Add the **Login Kit** and **Content Posting API** products. In Content Posting API, enable **Direct Post**.
3. Under Login Kit, choose the **Desktop** platform. Add the scopes `user.info.basic` and `video.publish`, and register this redirect URI:
   `http://localhost:8000/api/auth/tiktok/callback`
4. Paste the client key and secret into the panel and click **Connect**.

**Limits**
- Until TikTok audits your app, it can only post to a **private** account, with visibility **Only me**. TikTok rejects anything else with `unaudited_client_can_only_post_to_private_accounts`.
- To post publicly, submit the app for audit in the developer portal.
- TikTok's rules require the user to pick visibility (there is no default) and to see the Music Usage Confirmation line. The panel does both.

## Instagram Reels
Instagram only accepts direct file uploads through **Facebook Login**. The simpler "Instagram Login" needs the video at a public URL, which a local app can't provide.

1. Your Instagram must be a **Business or Creator** account, linked to a **Facebook Page** you manage.
2. On [Meta for Developers](https://developers.facebook.com/apps), create an app of type **Business**, and add **Facebook Login for Business**.
3. In its settings, add this to **Valid OAuth Redirect URIs**:
   `http://localhost:8000/api/auth/instagram/callback`
4. Optional: create a **login configuration** with these permissions, and paste its ID into the panel. Without one, Clip Studio requests the permissions directly.
   - `instagram_basic`
   - `instagram_content_publish`
   - `pages_show_list`
   - `pages_read_engagement`
   - `business_management`
5. Paste the App ID and App secret (App settings → Basic) and click **Connect**.

**Limits**
- In development mode, only people with a role on the app (you) can connect.
- The login lasts about 60 days; connect again when it expires.
- Instagram allows 100 API posts per 24 hours per account.

## Facebook Reels
Reels go to a **Facebook Page** you manage. Meta's API can't post to a personal profile.

1. On [Meta for Developers](https://developers.facebook.com/apps), create an app of type **Business** and add **Facebook Login for Business**. The app you made for Instagram works: add the same App ID and secret again as Facebook credentials.
2. In its settings, add this to **Valid OAuth Redirect URIs**:
   `http://localhost:8000/api/auth/facebook/callback`
3. Optional: create a **login configuration** with these permissions, and paste its ID into the panel. Without one, Clip Studio requests the permissions directly.
   - `pages_show_list`
   - `pages_read_engagement`
   - `pages_manage_posts`
   - `business_management`
4. Paste the App ID and App secret and click **Connect**. In Facebook's popup, tick the Page to post to.

**Limits**
- Reels must be **3 to 90 seconds** long and are posted publicly.
- The API takes no cover image, so Facebook picks one.
- In development mode, only people with a role on the app (you) can connect, and posts may be visible only to them until the app is live.
- The Page login doesn't expire on its own; connect again if you change your Facebook password or remove the app's access.

## Troubleshooting
- **"redirect_uri mismatch" or "invalid redirect":** the URI in the developer console must match exactly, including `localhost` (not `127.0.0.1`), the port, and the trailing path.
- **"isn't connected" / "login expired":** click Connect again.
- **The error message shows the platform's own response.** Search its error code in that platform's docs.
- **Untested against live accounts:** this was built from the platforms' API docs (November 2026) and tested against a mock of each API. It has not been run against real YouTube, TikTok, Instagram or Facebook accounts. If a platform changed something, the error text will say what.
