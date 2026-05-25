#include "bits/stdc++.h"
using namespace std;
#define all(x) (x).begin(),(x).end()
const int N = 1e5 + 2;
vector<int> e[N];
int rd[N], cd[N];
vector<int> ans;
void dfs(int u)
{
	while (e[u].size())
	{
		int v = e[u].back();
		e[u].pop_back();
		dfs(v);
		ans.push_back(v);
	}
}
int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	int n, m, i, x = 0, m0;
	cin >> n >> m; ans.reserve(m + 1); m0 = m;
	while (m--)
	{
		int u, v;
		cin >> u >> v;
		e[u].push_back(v);
		++cd[u]; ++rd[v];
	}
	for (i = 1; i <= n; i++) if (cd[i] != rd[i])
	{
		if (abs(cd[i] - rd[i]) > 1) goto no;
		++x;
	}
	if (x > 2) goto no; x = 0;
	for (i = 1; i <= n; i++) if (cd[i] > rd[i]) { x = i; break; }
	if (!x) for (i = 1; i <= n; i++) if (cd[i]) { x = i; break; }
	if (!x) x = 1;
	for (i = 1; i <= n; i++) sort(all(e[i])), reverse(all(e[i]));
	dfs(x); ans.push_back(x); reverse(all(ans));
	if (ans.size() != m0 + 1) goto no;
	for (i = 0; i < ans.size(); i++) cout << ans[i] << " \n"[i + 1 == ans.size()];
	return 0;
no:cout << "No" << endl;
}
