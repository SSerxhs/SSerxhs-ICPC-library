#include"bits/stdc++.h"
using namespace std;
//dengyaotriangle!

const int maxn = 1005;
const int maxq = 500005;
int n, m, q;
char a[maxn], b[maxn];
struct qryt {
    int x, nxt;
}z[maxq];
int qry[maxn][maxn];
int ans[maxq];
int r[maxn];
int bit[maxn];

int main() {
    ios::sync_with_stdio(0); cin.tie(0);
    cin >> q >> b >> a; n = strlen(a); m = strlen(b);
    //q,s,t
    for (int i = 1; i <= q; i++) {
        int a, b, c;
        cin >> a >> b >> c;
        if (a) {
            ans[i] = c - b;
            z[i].x = b; z[i].nxt = qry[a][c];
            qry[a][c] = i;
        }
    }
    for (int i = 0; i < n; i++)r[i] = i;
    for (int i = 0; i < m; i++) {
        int lp = -1;
        for (int j = 0; j < n; j++)if (a[j] == b[i]) { lp = j; break; }
        if (lp != -1) {
            for (int j = lp + 1; j < n; j++) {
                if (a[j] != b[i]) {
                    if (r[j - 1] < r[j])swap(r[j - 1], r[j]);
                }
            }
            for (int i = n - 1; i > lp; i--)r[i] = r[i - 1];
            r[lp] = -1;
        }
        for (int i = 0; i <= n; i++)bit[i] = 0;
        for (int j = 0; j < n; j++) {
            if (r[j] != -1) {
                for (int p = n - r[j]; p <= n; p += p & -p)bit[p]++;
            }
            for (int y = qry[i + 1][j + 1]; y; y = z[y].nxt) {
                for (int p = n - z[y].x; p; p -= p & -p)ans[y] -= bit[p];
            }
        }
    }
    for (int i = 1; i <= q; i++)cout << ans[i] << '\n';
    return 0;
}
