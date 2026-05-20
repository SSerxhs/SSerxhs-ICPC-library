#include "bits/stdc++.h"
using namespace std;
const int N = 152;
vector<int> e[N];
int lk[N], kl[N], ed[N], now;
bool dfs(int u)
{
	for (int v : e[u]) if (ed[v] != now)
	{
		ed[v] = now;
		if (!lk[v] || dfs(lk[v])) return lk[v] = u;
	}
	return 0;
}
int main()
{
	int n, m, i;
	ios::sync_with_stdio(0); cin.tie(0);
	cin >> n >> m;
	while (m--)
	{
		int u, v;
		cin >> u >> v;
		e[u].push_back(v);
	}
	int r = 0;
	for (i = 1; i <= n; i++) r += dfs(now = i);
	for (i = 1; i <= n; i++) kl[lk[i]] = i;
	for (i = 1; i <= n; i++) if (ed[i] != -1 && !lk[i])
	{
		vector<int> ans;
		int u = i;
		while (u)
		{
			ed[u] = -1;
			ans.push_back(u);
			u = kl[u];
		}
		for (int j = 0; j < ans.size(); j++) cout << ans[j] << " \n"[j + 1 == ans.size()];
	}
	cout << n - r << endl;
}
