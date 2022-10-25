#include <bits/stdc++.h>
using namespace std;
const int N=1e5+2,B=126271,lim=1e9;
int a[N];
int main()
{
	ios::sync_with_stdio(0);cin.tie(0);
	int t=1,n=1e5,i;
	cout<<t<<'\n'<<n<<'\n';
	for (i=1;i<=20;i++) a[i]=i;
	for (i=21;i<=n;i++) a[i]=a[i-20]+B;
	assert(a[n]<=lim);
	for (i=1;i<=n;i++) cout<<a[i]<<" \n"[i==n];
}