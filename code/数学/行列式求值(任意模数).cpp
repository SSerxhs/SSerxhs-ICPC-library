#include "bits/stdc++.h"
using namespace std;
using ll = long long;
const int N = 502, p = 998244353;
int cal(int a[][N], int n)
{
	int i, j, k, r = 1, fh = 0, l;
	for (i = 1; i <= n; i++)
	{
		k = i;
		for (j = i + 1; j <= n; j++) if (a[j][i]) { k = j; break; }
		if (a[k][i] == 0) return 0;
		if (i != k) { swap(a[k], a[i]); fh ^= 1; }
		for (j = i + 1; j <= n; j++)
		{
			if (a[j][i] > a[i][i]) swap(a[j], a[i]), fh ^= 1;
			while (a[j][i])
			{
				l = a[i][i] / a[j][i];
				for (k = i; k <= n; k++) a[i][k] = (a[i][k] + (ll)(p - l) * a[j][k]) % p;
				swap(a[j], a[i]); fh ^= 1;
			}
		}
		r = (ll)r * a[i][i] % p;
	}
	if (fh) return (p - r) % p;
	return r;
}
int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	int n, i, j;
	static int a[N][N];
	cin >> n;
	for (i = 1; i <= n; i++) for (j = 1; j <= n; j++) cin >> a[i][j];
	cout << cal(a, n) << endl;
}

