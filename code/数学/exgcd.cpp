int exgcd(int a, int b, int c)//ax+by=c,return x
{
	if (a == 0) return c / b;
	return (c - (ll)b * exgcd(b % a, a, c)) / a % b;
}

