package com.company;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.Stack;
import java.util.TreeMap;
import java.util.TreeSet;

public class Main {

    public static void main(String[] args) {
        // 1. BFS
        // 2. DFS
        // 3. Union Find
        // 4. Topological sort
        // System.out.println(numberOfOddSubarrays(new int[]{2, 2, 2, 1, 2, 2, 1, 2, 2, 2}, 2));
        //  System.out.println(numSubarraysWithSum(new int[]{0,0,0,0,0}, 0));
        //System.out.println(minSubArrayLen(213 ,new int[]{12,28,83,4,25,26,25,2,25,25,25,12}));
//        List<List<Integer>> l = new ArrayList<>();
//        l.add(Arrays.asList(1,3));
//        l.add(Arrays.asList(3,0,1));
//        l.add(Arrays.asList(2));
//        l.add(Arrays.asList());
        //[[1,0,0,1],[0,1,1,0],[0,1,1,1],[1,0,1,1]]

        int[][] multi = new int[][]{
            {0, 0, 1, 0},
            {1, 0, 1, 0},
            {0, 1, 1, 0},
            {0, 0, 0, 0}
        };

        // [A,-1,0,A]
        // [A,A,A,-1]
        // [A,-1,A,-1]
        // [0,-1,A,A]]

        char[][] ccc = new char[][]{
            {'O', 'O'}, {'O', 'O'}
        };

        int[][] d = new int[][]{
            {2147483647, -1, 0, 2147483647},
            {2147483647, 2147483647, 2147483647, -1},
            {2147483647, -1, 2147483647, -1},
            {0, -1, 2147483647, 2147483647}
        };
        int[] a = new int[]{
            3, 1, 3, 4, 2
        };
        int[][] n = new int[][]{
            {0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0},
            {0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0},
            {0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0}
        };

        int[][] nn = new int[][]{
            {1, 1, 0, 0, 0},
            {1, 1, 0, 0, 0},
            {0, 0, 0, 1, 1},
            {0, 0, 0, 1, 1}
        };

        System.out.println(numDistinctIslands(nn));
    }

    public static Set<String> set = new HashSet<>();
    public static String curStr = "";

    public static int numDistinctIslands(int[][] grid) {
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[i].length; j++) {
                if (grid[i][j] == 1) {
                    curStr = "";
                    numDistinctIslandsDfs(grid, i, j, 0,0);
                    if (!set.contains(curStr)) {
                        set.add(curStr);
                    }
                }
            }
        }

        return set.size();
    }

    public static void numDistinctIslandsDfs(int[][] grid, int i, int j, int ilocal, int jlocal) {
        if (i < 0 || j < 0 || i > grid.length - 1 || j > grid[i].length - 1 || grid[i][j] == 0) {
            return;
        }

        curStr += ilocal;
        curStr += jlocal;

        grid[i][j] = 0;

        numDistinctIslandsDfs(grid, i + 1, j, ilocal + 1, jlocal);
        numDistinctIslandsDfs(grid, i - 1, j, ilocal - 1, jlocal);
        numDistinctIslandsDfs(grid, i, j + 1, ilocal, jlocal + 1);
        numDistinctIslandsDfs(grid, i, j - 1, ilocal, jlocal - 1);
    }

    public static int maxAreaOfIslandSize = 0;
    public static int currentCount = 0;

    public static int maxAreaOfIsland(int[][] grid) {

        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[i].length; j++) {
                if (grid[i][j] == 1) {
                    currentCount = 0;
                    maxAreaOfIslandDfs(grid, i, j);
                }
            }
        }

        return maxAreaOfIslandSize;
    }

    public static void maxAreaOfIslandDfs(int[][] grid, int i, int j) {
        if (i < 0 || j < 0 || i > grid.length - 1 || j > grid[i].length - 1 || grid[i][j] == 0) {
            return;
        }
        currentCount++;
        grid[i][j] = 0;
        maxAreaOfIslandSize = Math.max(maxAreaOfIslandSize, currentCount);

        maxAreaOfIslandDfs(grid, i + 1, j);
        maxAreaOfIslandDfs(grid, i - 1, j);
        maxAreaOfIslandDfs(grid, i, j + 1);
        maxAreaOfIslandDfs(grid, i, j - 1);
    }

    public static int earliestAcqCount;

    public static int earliestAcq(int[][] logs, int n) {
        int[] parents = new int[n];
        Arrays.sort(logs, (a, b) -> Integer.compare(a[0], b[0]));
        for (int i = 0; i < n; i++) {
            parents[i] = i;
        }
        earliestAcqCount = n;

        for (int i = 0; i < logs.length; i++) {
            int x = earliestAcqFindRoot(parents, logs[i][1]);
            int y = earliestAcqFindRoot(parents, logs[i][2]);
            if (x != y) {
                earliestAcqCount--;
                if (earliestAcqCount == 1) {
                    return logs[i][0];
                }
                parents[y] = x;
            }
        }

        return -1;
    }

    public static int earliestAcqFindRoot(int[] parent, int node) {
        return parent[node] == node ? node : earliestAcqFindRoot(parent, parent[node]);
    }

    public static int findDuplicate(int[] nums) {
        int[] arr = new int[nums.length + 1];

        for (int i = 0; i < nums.length; i++) {
            if (arr[nums[i]] != 0) {
                return nums[i];
            }
            arr[nums[i]] = nums[i];
        }

        return -1;
    }

    public static void wallsAndGates(int[][] rooms) {

        for (int i = 0; i < rooms.length; i++) {
            for (int j = 0; j < rooms[0].length; j++) {
                if (rooms[i][j] == 0) {
                    wallsAndGatesDfs(rooms, i, j, 0);
                }
            }
        }
        int t = 0;
    }

    public static void wallsAndGatesDfs(int[][] rooms, int i, int j, int distance) {
        if (i < 0 || i >= rooms.length || j < 0 || j >= rooms[0].length || (rooms[i][j] <= distance && distance != 0)) {
            return;
        }

        if (rooms[i][j] == -1) {
            return;
        }

        if (distance <= rooms[i][j]) {

            rooms[i][j] = distance;
            wallsAndGatesDfs(rooms, i + 1, j, distance + 1);
            wallsAndGatesDfs(rooms, i - 1, j, distance + 1);
            wallsAndGatesDfs(rooms, i, j + 1, distance + 1);
            wallsAndGatesDfs(rooms, i, j - 1, distance + 1);
        }
    }

    public static void solve(char[][] board) {
        if (board.length == 0) {
            return;
        }

        for (int i = 0; i < board.length; i++) {
            if (board[i][0] == 'O') {
                solveDfs(board, i, 0);
            }
            if (board[i][board[0].length - 1] == 'O') {
                solveDfs(board, i, board[0].length - 1);
            }
        }
        for (int i = 0; i < board[0].length; i++) {
            if (board[0][i] == 'O') {
                solveDfs(board, 0, i);
            }
            if (board[board.length - 1][i] == 'O') {
                solveDfs(board, board.length - 1, i);
            }
        }

        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                if (board[i][j] == '!') {
                    board[i][j] = 'O';
                } else {
                    board[i][j] = 'X';
                }
            }
        }
        int tt = 0;
    }

    public static void solveDfs(char[][] board, int i, int j) {
        //out of range
        if (i < 0 || j < 0 || i > board.length - 1 || j > board[i].length - 1 || board[i][j] == 'X' || board[i][j] == '!') {
            return;
        }
        board[i][j] = '!';
        solveDfs(board, i + 1, j);
        solveDfs(board, i - 1, j);
        solveDfs(board, i, j + 1);
        solveDfs(board, i, j - 1);
    }

    public static int singleNodeCount;

    public static int countComponents(int n, int[][] edges) {
        singleNodeCount = n;
        int[] parent = new int[n];
        for (int i = 0; i < n; i++) {
            parent[i] = -1;
        }

        for (int i = 0; i < edges.length; i++) {
            countComponentsUnion(parent, edges[i][0], edges[i][1]);
        }

        return singleNodeCount;
    }

    public static void countComponentsUnion(int[] parent, int a, int b) {
        int aRoot = countComponentsFind(parent, a);
        int bRoot = countComponentsFind(parent, b);
        if (aRoot != bRoot) {
            singleNodeCount--;
            parent[aRoot] = bRoot;
        }
    }

    // parent [-1,-1,-1,-1,-1]
    // 0, 1, 2, 3, 4
    public static int countComponentsFind(int[] parent, int node) {
        if (parent[node] == -1) {
            return node;
        }
        return countComponentsFind(parent, parent[node]);
    }

    public static int findCircleNumUnionFind(int[][] isConnected) {
        int res = 0;

        int[] parent = new int[isConnected.length];
        for (int i = 0; i < parent.length; i++) {
            parent[i] = -1;
        }

        for (int i = 0; i < isConnected.length; i++) {
            for (int j = 0; j < isConnected[0].length; j++) {
                if (isConnected[i][j] == 1) {
                    findCircleNumUnion(parent, i, j);
                }
            }
        }

        for (int i = 0; i < parent.length; i++) {
            if (parent[i] == -1) {
                res++;
            }
        }

        return res;
    }

    public static void findCircleNumUnion(int[] parent, int i, int j) {
        int iRoot = circleNumFindRoot(parent, i);
        int jRoot = circleNumFindRoot(parent, j);

        if (iRoot != jRoot) {
            parent[iRoot] = jRoot;
        }

    }

    public static int circleNumFindRoot(int[] parent, int node) {
        if (parent[node] == -1) {
            return node;
        }
        return circleNumFindRoot(parent, parent[node]);
    }

    public static int countServers(int[][] grid) {
        if (grid.length == 0) {
            return 0;
        }
        int serverCounter = 0;

        int[] rowCount = new int[grid.length];
        // [0,0,0,0,0]
        int[] columnCount = new int[grid[0].length];
        // [0,0,0,0,0]
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1) {
                    rowCount[i]++;
                    columnCount[j]++;
                    serverCounter++;
                }
            }
        }

        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1 && rowCount[i] == 1 && columnCount[j] == 1) {
                    serverCounter--;
                }
            }
        }

        return serverCounter;
    }

    public static int longestConsecutive(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        int maxRes = 1;
        PriorityQueue<Integer> q = new PriorityQueue<>(nums.length, (a, b) -> {
            return a.compareTo(b);
        });

        for (int i = 0; i < nums.length; i++) {
            q.add(nums[i]);
        }

        int len = 1;

        int el = q.poll();
        for (int i = 0; i < nums.length - 1; i++) {

            int curr = q.poll();
            while (curr == el && !q.isEmpty()) {
                curr = q.poll();
                i++;
            }
            if (el + 1 == curr) {
                len++;
                maxRes = Math.max(maxRes, len);
            } else {
                len = 1;
            }
            el = curr;
        }

        return maxRes;
    }

    public static int closedIsland(int[][] grid) {

        int res = 0;

        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[i].length; j++) {
                if (i == 0 || j == 0 || i == grid.length - 1 || j == grid[0].length - 1) {
                    closedIslandDfs(grid, i, j);
                }
            }
        }

        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[i].length; j++) {
                if (grid[i][j] == 0) {
                    res++;
                    closedIslandDfs(grid, i, j);
                }
            }
        }

        return res;
    }

    public static void closedIslandDfs(int[][] grid, int i, int j) {
        if (i < 0 || j < 0 || i >= grid.length || j >= grid[i].length || grid[i][j] == 1) {
            return;
        }
        grid[i][j] = 1;
        closedIslandDfs(grid, i + 1, j);
        closedIslandDfs(grid, i - 1, j);
        closedIslandDfs(grid, i, j + 1);
        closedIslandDfs(grid, i, j - 1);
    }

    public static int numEnclaves(int[][] A) {
        int res = 0;

        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < A[i].length; j++) {
                // only check the outer positions
                if (i == 0 || j == 0 || i == A.length - 1 || j == A[i].length - 1) {
                    numEnclavesDFS(A, i, j);
                }
            }
        }

        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < A[i].length; j++) {
                if (A[i][j] == 1) {
                    res++;
                }
            }
        }

        return res;
    }

    public static void numEnclavesDFS(int[][] A, int i, int j) {
        if (i >= 0 && j >= 0 && i < A.length && j < A[i].length && A[i][j] == 1) {
            A[i][j] = 0;
            numEnclavesDFS(A, i + 1, j);
            numEnclavesDFS(A, i - 1, j);
            numEnclavesDFS(A, i, j + 1);
            numEnclavesDFS(A, i, j - 1);
        }
    }

    public static int[] parentServer;
    public static int[] rankServer;
    public static int serverResult = 0;

    public static int[] arrResult;

    public static int countServerss(int[][] grid) {
        var copyGrid = grid.clone();

        parentServer = new int[grid.length];
        arrResult = new int[grid.length];
        int arrRes = 0;

        serverResult = grid.length;
        for (int i = 0; i < grid.length; i++) {
            parentServer[i] = i;
        }

        for (int i = 0; i < grid.length - 1; i++) {
            for (int j = i + 1; j < grid[i].length; j++) {
                if (grid[i][j] == 1) {
                    copyGrid[i][j] = -1;
                    copyGrid[i][i] = -1;
                    // unionServers(i, j);
                }
            }
        }
        for (int i = 0; i < grid.length; i++) {

        }
        return grid.length - arrRes;
    }

    private static void unionServers(int a, int b) {
        // find root of a&b
        var aRoot = findRoot(a);
        var bRoot = findRoot(b);

        // if they are the same, do nothing
        if (aRoot == bRoot) {
            return;
        }

        parentServer[b] = a;
        arrResult[a] = -1;
        arrResult[b] = -1;

        serverResult--;
    }

    private static int findRoot(int root) {
        while (root != parentServer[root]) {
            parentServer[root] = parentServer[parentServer[root]];
            root = parentServer[root];
        }
        return root;
    }

    private static int[] parent, rank;
    private static int count = 0;

    public static int findCircleNum(int[][] isConnected) {
        int n = isConnected.length;

        count = n;
        parent = new int[n];
        rank = new int[n];

        //every node is parent to itself
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }

        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                if (isConnected[i][j] == 1) {
                    union(i, j);
                }
            }
        }

        return count;
    }

    public static void union(int p, int q) {
        // find roots
        int rootP = find(p);
        int rootQ = find(q);
        if (rootP == rootQ) {
            return; // if it is loop to itself, skip it
        }

        if (rank[rootQ] > rank[rootP]) {
            parent[rootP] = rootQ;
        } else {
            parent[rootQ] = rootP;
            if (rank[rootP] == rank[rootQ]) {
                rank[rootP]++;
            }
        }
        count--;
    }

    public static int find(int p) {
        while (p != parent[p]) {
            parent[p] = parent[parent[p]];
            p = parent[p];
        }
        return p;
    }

    public static boolean canVisitAllRooms(List<List<Integer>> rooms) {
        var visited = new boolean[rooms.size()];
        visited[0] = true;
        Set<Integer> set = new HashSet<>();

        for (int i = 0; i < rooms.size(); i++) {
            if (!visited[i]) {
                return false;
            }
            rec(rooms, i, visited, set);
        }

        return true;
    }

    public static void rec(List<List<Integer>> rooms, int room, boolean[] visited, Set<Integer> set) {

        set.addAll(rooms.get(room));
        for (int i = 0; i < rooms.get(room).size(); i++) {
            if (visited[rooms.get(room).get(i)]) {
                continue;
            }
            visited[rooms.get(room).get(i)] = true;
            rec(rooms, rooms.get(room).get(i), visited, set);
        }
    }

    public static int numberOfOddSubarrays(int[] nums, int k) {
        int start = 0, end = 0, oddCount = 0, count = 0, res = 0;

        while (end < nums.length) {
            int num = nums[end];
            if (num % 2 != 0) {
                oddCount++;
                count = 0;
            }

            while (oddCount == k) {
                int num1 = nums[start];
                if (num1 % 2 != 0) {
                    oddCount--;
                }
                count++;
                start++;
            }
            end++;
            res += count;
        }

        return res;
    }

    public static int maxVowels(String s, int k) {
        int start = 0, end = 0, vowelClount = 0, countMax = 0, charCount = 0;

        while (end < s.length()) {

            var c = s.charAt(end);
            if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') {
                vowelClount++;
            }
            charCount++;
            if (charCount == k) {

                countMax = Math.max(countMax, vowelClount);
                charCount--;
                var c1 = s.charAt(start);
                if (c1 == 'a' || c1 == 'e' || c1 == 'i' || c1 == 'o' || c1 == 'u') {
                    vowelClount--;
                }

                start++;
            }
            end++;
        }

        return countMax;
    }

    public static int numberOfSubstrings(String s) {
        Map<Character, Integer> map = new HashMap<>();
        int start = 0, end = 0, sum = 0, res = 0;

        //ab cab c
        // a - 1
        // b - 1
        // c - 1

        // count -  3
        // start = 2
        // res = 1
        while (end < s.length()) {
            var c = s.charAt(end);
            map.put(c, map.getOrDefault(c, 0) + 1);

            while (map.get('a') > 0 && map.get('b') > 0 && map.get('c') > 0) {
                // decrement start from map
                var c1 = s.charAt(start);
                map.put(c1, map.getOrDefault(c1, 0) - 1);
                if (map.get(c1) == 0) {
                    map.remove(c1);
                }

                start++;
                sum++;
            }
            end++;
            res += sum;
        }
        return res;
    }

    public static int longestSubarray(int[] nums) {
        LinkedList<Integer> ad = new LinkedList<>();

        int zeroCounter = 0, max = -1;
        for (int i = 0; i < nums.length; i++) {
            int num = nums[i];
            ad.add(num);
            if (num == 0) {
                zeroCounter++;
            }
            if (zeroCounter > 1) {
                var item = ad.removeFirst();
                if (item == 0) {
                    zeroCounter--;
                }

            } else {
                max = Math.max(max, ad.size());
            }

        }
        return max - 1;
    }

    public static int longestOnes(int[] nums, int k) {
        ArrayDeque<Integer> list = new ArrayDeque<>();
        int zeroCounter = 0, max = 0;

        for (int i = 0; i < nums.length; i++) {
            int num = nums[i];
            list.add(nums[i]);
            if (num == 0) {
                zeroCounter++;
            }
            if (zeroCounter > k) {
                int pooled = list.pollFirst();
                if (pooled == 0) {
                    zeroCounter--;
                }
            } else { // valid, check if queue.len > max
                max = Math.max(max, list.size());
            }

        }
        return max;
    }

    public static int minimumCardPickup(int[] cards) {
        // store the last value
        // if it exiist always compare it with existing
        int minRes = Integer.MAX_VALUE;
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < cards.length; i++) {
            int card = cards[i];
            if (map.containsKey(card)) {
                minRes = Math.min(minRes, i - map.get(card) + 1);
            }
            map.put(card, i);
        }

        return minRes == Integer.MAX_VALUE ? -1 : minRes;
    }

    public static int lengthOfLongestSubstring(String s) {
        Map<Character, Integer> map = new HashMap<>();
        int start = 0, end = 0, count = 0, maxLen = 0;
        // abcabcbb

        // p - 1
        // w - 1
        // k - 1
        // e -
        while (end < s.length()) {
            Character c = s.charAt(end);
            if (map.containsKey(c) && map.get(c) > 0) {
                count++;
            }
            map.put(c, map.getOrDefault(c, 0) + 1);

            while (count > 0) {
                Character cStart = s.charAt(start);
                map.put(cStart, map.getOrDefault(cStart, 0) - 1);
                start++;
                if (map.get(cStart) > 0) {
                    count--;
                }
            }
            end++;

            if (end - start > maxLen) {
                maxLen = end - start;
            }
        }

        return maxLen;
    }

    public static String minWindow(String s, String t) {
        Map<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < t.length(); i++) {
            map.put(t.charAt(i), map.getOrDefault(t.charAt(i), 0) + 1);
        }
        for (int i = 0; i < s.length(); i++) {
            if (map.containsKey(s.charAt(i))) {
                continue;
            }
            map.put(s.charAt(i), 0);
        }

        int start = 0, end = 0, minStart = 0, minLen = Integer.MAX_VALUE, counter = t.length();

        while (end < s.length()) {
            Character c = s.charAt(end);
            if (map.get(c) > 0) {
                counter--;
            }

            map.put(c, map.getOrDefault(c, 0) - 1);
            end++;

            while (counter == 0) {
                if (minLen > end - start) {
                    minLen = end - start;
                    minStart = start;
                }

                Character c1 = s.charAt(start);
                map.put(c1, map.getOrDefault(c1, 0) + 1);

                if (map.get(c1) > 0) {
                    counter++;
                }

                start++;
            }
        }

        return minLen != Integer.MAX_VALUE ? s.substring(minStart, minStart + minLen) : "";
    }

    public static int minSubArrayLen(int target, int[] nums) {
        return 0;
    }

    public static int minSetSize(int[] arr) {
        Map<Integer, Integer> map = new TreeMap<Integer, Integer>();
        int sum = arr.length, result = 0;
        for (int i = 0; i < arr.length; i++) {
            map.put(arr[i], map.getOrDefault(arr[i], 0) + 1);
        }
        PriorityQueue<Integer> pq = new PriorityQueue<>((a, b) -> b - a);
        for (var key : map.keySet()) {
            pq.add(map.get(key));
        }

        while (pq.size() != 0) {
            sum -= pq.poll();
            result++;
            if (sum <= arr.length / 2) {
                break;
            }
        }

        return result;
    }

    public int minAddToMakeValid(String s) {
        Stack<Character> stack = new Stack<>();
        int res = 0, i = 0;
        while (i < s.length()) {
            if (s.charAt(i) == '(') {
                stack.add('(');
                res++;
            } else {
                if (!stack.empty() && s.charAt(i) == ')') {
                    stack.pop();
                    res--;
                } else {
                    res++;
                }
            }
            i++;
        }
        return res;
    }

    public static int minSwaps(String s) {
        int errorCount = 0;

        Stack<Character> stack = new Stack<>();

        for (int i = 0; i < s.length(); i++) {
            var c = s.charAt(i);
            if (c == '[') {
                stack.add(c);
            } else {
                if (!stack.empty()) {
                    stack.pop();
                } else {
                    errorCount++;
                }
            }
        }

        return (errorCount + 1) / 2;

    }

    public static boolean carPooling(int[][] trips, int capacity) {

        int arrLen = 0;
        for (int i = 0; i < trips.length; i++) {
            arrLen = Math.max(arrLen, trips[i][2]);
        }

        int[] stops = new int[arrLen + 1];

        for (int i = 0; i < trips.length; i++) {
            for (int j = trips[i][1]; j < trips[i][2]; j++) {
                //stops =  0 0 0 0 0

                //stops =  2,2,2,2,2,0,0
                //trips =  1,2,3,4,5

                //stops =  2,2,5,2,2,0,0
                //trips =  3,4,5,6,7

                stops[j] += trips[i][0];
                if (stops[j] > capacity) {
                    return false;
                }
            }
        }

        return true;
    }

    public static List<Integer> partitionLabels(String s) {
        var result = new ArrayList<Integer>();
        int[] rightMaxValues = new int[26];
        for (int i = 0; i < s.length(); i++) {
            rightMaxValues[s.charAt(i) - 'a'] = i;
        }

        int groupMaxRight = -1;
        int groupMaxLeft = 0;
        for (int i = 0; i < s.length(); i++) {
            groupMaxRight = Math.max(groupMaxRight, rightMaxValues[s.charAt(i) - 'a']);
            if (groupMaxRight == i) {
                groupMaxRight = i + 1;
                result.add(groupMaxRight - groupMaxLeft);
                groupMaxLeft = i + 1;

            }
        }

        return result;
    }

    public static int canCompleteCircuit(int[] gas, int[] cost) {
        return 1;
    }

    //1432219   k=3
    public static String removeKdigits(String num, int k) {
        if (k == num.length()) {
            return "0";
        }
        Stack<Character> stack = new Stack<>();

        int i = 0;
        while (i < num.length()) {
            while (k > 0 && !stack.empty() && stack.peek() > num.charAt(i)) {
                stack.pop();
                k--;
            }
            stack.add(num.charAt(i));
            i++;
        }

        // edge case 1111

        while (k > 0) {
            stack.pop();
            k--;
        }

        // 0012
        StringBuilder sb = new StringBuilder();

        while (!stack.empty()) {
            sb.append(stack.pop());
        }

        sb.reverse();
        while (sb.length() > 1 && sb.charAt(0) == '0') {
            sb.deleteCharAt(0);
        }

        return sb.toString();

    }

    public static int jump(int[] nums) {
        int result = 0, end = 0, potentialJump = 0;

        for (int i = 0; i < nums.length; i++) {
            potentialJump = Math.max(i + nums[i], potentialJump);
            if (i == end) {
                end = potentialJump;
                result++;
            }
        }

        return result;
    }

    public static int candy(int[] ratings) {

        int[] result = new int[ratings.length];
        for (int i = 0; i < ratings.length; i++) {
            result[i] = 1;
        }

        for (int i = 1; i < ratings.length; i++) {
            if (ratings[i - 1] < ratings[i]) {
                result[i] = ratings[i - 1] + 1;
            }
        }

        for (int i = ratings.length - 1; i > 0; i--) {
            if (ratings[i] > ratings[i + 1]) {
                if (result[i] < result[i + 1] + 1) {
                    result[i] = result[i + 1] + 1;
                }
            }
        }
        int sum = 0;
        for (int i = 0; i < result.length; i++) {
            sum += result[i];
        }
        return sum;
    }

    public static boolean canJump(int[] nums) {

        int daysTotal = 0;
        for (int i = 0; i < nums.length; i++) {
            if (i > daysTotal) {
                return false;
            }
            daysTotal = Math.max(daysTotal, i + nums[i]);
        }

        return true;
    }

    // TLE ;(
    public static boolean canJumpDfs(int[] nums, int pos) {
        if (pos >= nums.length - 1) {
            return true;
        }
        for (int i = pos; i < nums[pos] + pos; i++) {
            if (!canJumpDfs(nums, i + 1)) {
                continue;
            } else {
                return true;
            }

        }

        return false;
    }

    public static boolean makesquare(int[] matchsticks) {
        int sum = 0;
        for (int i = 0; i < matchsticks.length; i++) {
            sum += matchsticks[i];
        }
        if (sum % 4 != 0) {
            return false;
        }
        boolean[] used = new boolean[matchsticks.length];
        Arrays.sort(matchsticks);
        return makesquareDfs(matchsticks, used, sum / 4, 0, 0, 4);
    }

    static boolean makesquareDfs(int[] matchsticks, boolean[] used, int sum, int curSum, int index, int sides) {
        if (sides == 1) {
            return true;
        }
        if (curSum > sum) {
            return false;
        }
        if (curSum == sum) {
            return makesquareDfs(matchsticks, used, sum, 0, 0, sides - 1);
        }
        for (int i = index; i < matchsticks.length; i++) {

            if (used[i]) {
                continue;
            }
            used[i] = true;
            curSum += matchsticks[i];
            if (makesquareDfs(matchsticks, used, sum, curSum, index, sides)) {
                return true;
            }
            used[i] = false;
            curSum -= matchsticks[i];
        }

        return false;
    }

    public static boolean canPartitionKSubsets(int[] nums, int k) {
        boolean[] used = new boolean[nums.length];
        int sum = 0;
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
        }
        if (sum % k != 0) {
            return false;
        }

        return canPartitionKSubsetsDfs(nums, used, k, sum / k, 0, 0);
    }

    static boolean canPartitionKSubsetsDfs(int[] nums, boolean[] used, int k, int sum, int curSum, int index) {
        if (curSum > sum) {
            return false;
        }
        if (k == 1) {
            return true;
        }
        if (curSum == sum) {
            return canPartitionKSubsetsDfs(nums, used, k - 1, sum, 0, 0);
        }

        for (int i = index; i < nums.length; i++) {
            if (used[i]) {
                continue;
            }
            used[i] = true;
            if (canPartitionKSubsetsDfs(nums, used, k, sum, curSum + nums[i], i + 1)) {
                return true;
            }
            used[i] = false;
        }

        return false;
    }

    public static List<List<Integer>> combinationSum2(int[] candidates, int target) {
        List<List<Integer>> result = new ArrayList<>();
        boolean[] used = new boolean[candidates.length];
        Arrays.sort(candidates);
        combinationSum2Dfs(result, candidates, new ArrayList<>(), target, used, 0, 0);
        return result;
    }

    static void combinationSum2Dfs(List<List<Integer>> result,
                                   int[] candidates,
                                   List<Integer> currentList,
                                   int target,
                                   boolean[] used, int sum, int index) {

        if (sum == target) {
            result.add(new ArrayList<>(currentList));
            return;
        }

        if (sum > target) {
            return;
        }

        for (int i = index; i < candidates.length; i++) {
            if (used[i]) {
                continue;
            }
            if (i > index && candidates[i] == candidates[i - 1]) {
                continue;
            }
            used[i] = true;
            currentList.add(candidates[i]);
            sum += candidates[i];
            combinationSum2Dfs(result, candidates, currentList, target, used, sum, i + 1);
            sum -= candidates[i];
            used[i] = false;
            currentList.remove(currentList.size() - 1);
        }
    }

    static int maxLength = 0;

    public static int maxLength(List<String> arr) {
        if (arr == null || arr.size() == 0) {
            return 0;
        }

        maxLengthDfs(arr, "", 0);
        return maxLength;
    }

    static void maxLengthDfs(List<String> arr, String currentStr, int index) {
        if (itIsUnique(currentStr)) {
            maxLength = Math.max(maxLength, currentStr.length());
        } else {
            return;
        }
        if (index == arr.size()) {
            return;
        }

        for (int i = index; i < arr.size(); i++) {
            maxLengthDfs(arr, currentStr + arr.get(i), i + 1);
        }
    }

    static boolean itIsUnique(String str) {
        HashSet<Character> hash = new HashSet<>();

        for (int i = 0; i < str.length(); i++) {
            if (hash.contains(str.charAt(i))) {
                return false;
            }
            hash.add(str.charAt(i));
        }
        return true;
    }

    public static List<List<Integer>> findSubsequences(int[] nums) {
        HashSet<List<Integer>> result = new HashSet<>();
        boolean[] visited = new boolean[nums.length];

        findSubsequencesDfs(result, nums, new ArrayList<>(), visited, 0);
        return new ArrayList<>(result);
    }

    static void findSubsequencesDfs(HashSet<List<Integer>> result,
                                    int[] nums,
                                    List<Integer> currentSubsequence,
                                    boolean[] visited,
                                    int index) {
        if (currentSubsequence.size() >= 2) {
            result.add(new ArrayList<>(currentSubsequence));
        }
        for (int i = index; i < nums.length; i++) {
            if (visited[i]) {
                continue;
            }
            if (i > index && nums[i - 1] == nums[i] && visited[i - 1] == false) {
                continue;
            }

            if (currentSubsequence.size() > 0 && currentSubsequence.get(currentSubsequence.size() - 1) > nums[i]) {
                continue;
            }

            currentSubsequence.add(nums[i]);
            visited[i] = true;
            findSubsequencesDfs(result, nums, currentSubsequence, visited, i + 1);
            currentSubsequence.remove(currentSubsequence.size() - 1);
            visited[i] = false;

        }
    }

    static public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        boolean[] used = new boolean[nums.length];
        permuteUniqueDfs(result, nums, new ArrayList<>(), used);
        return result;
    }

    static void permuteUniqueDfs(List<List<Integer>> result, int[] nums, List<Integer> currentArray, boolean[] used) {
        if (currentArray.size() == nums.length) {
            result.add(new ArrayList<>(currentArray));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (used[i]) {
                continue;
            }
            if (i > 0 && nums[i - 1] == nums[i] && !used[i - 1]) {
                continue;
            }
            currentArray.add(nums[i]);
            used[i] = true;
            permuteUniqueDfs(result, nums, currentArray, used);
            used[i] = false;
            currentArray.remove(currentArray.size() - 1);
        }
    }

    static boolean isPalindrome(String str) {
        int left = 0, right = str.length() - 1;
        while (left < right) {
            if (str.charAt(left) != str.charAt(right)) {
                return false;
            }
        }
        return true;
    }

    static int maxFindMaxForm = 0;

    public static int findMaxForm(String[] strs, int m, int n) {

        boolean[] used = new boolean[strs.length];
        int[] zeros = new int[strs.length];
        int[] ones = new int[strs.length];
        findMaxFormDfs(strs, m, n, 0, used, 0, zeros, ones);
        return maxFindMaxForm;
    }

    // TODO -> optimize
    static void findMaxFormDfs(String[] strs,
                               int m,
                               int n,
                               int index,
                               boolean[] used,
                               int len,
                               int[] zeros,
                               int[] ones) {

        if (m < 0 && n < 0) {
            return;
        } else {
            maxFindMaxForm = Math.max(len, maxFindMaxForm);
            for (int i = index; i < strs.length; i++) {
                if (used[i]) {
                    continue;
                }
                used[i] = true;
                int mm = m, nn = n;
                if (zeros[i] != 0 && ones[i] != 0) {
                    mm = zeros[i];
                    nn = ones[i];
                } else {

                    for (int c = 0; c < strs[i].length(); c++) {
                        if (strs[i].charAt(c) == '0') {
                            mm--;
                        }
                        if (strs[i].charAt(c) == '1') {
                            nn--;
                        }
                    }
                    zeros[i] = mm;
                    ones[i] = nn;
                }
                findMaxFormDfs(strs, mm, nn, i + 1, used, len + 1, zeros, ones);
                used[i] = false;
            }
        }
    }

    public static String findDifferentBinaryString(String[] nums) {
        int n = nums[0].length();
        if (n == 1 && nums[0].length() == 1 && nums[0].charAt(0) == '0') {
            return "1";
        }
        if (n == 1 && nums[0].length() == 1 && nums[0].charAt(0) == '1') {
            return "0";
        }
        HashSet<String> hashSet = new HashSet<>(Arrays.asList(nums));
        return dfsFindDifferentBinaryString(hashSet, new ArrayList<>(), n, '0');
    }

    static String dfsFindDifferentBinaryString(HashSet<String> hashSet,
                                               List<Character> tempList,
                                               int n,
                                               Character currentChar) {
        if (tempList.size() == n) {

            StringBuilder builder = new StringBuilder(tempList.size());
            for (Character ch : tempList) {
                builder.append(ch);
            }
            var t = builder.toString();
            if (!hashSet.contains(t)) {
                return t;
            }
        }
        if (tempList.size() > n) {
            return "";
        }
        for (int i = 0; i < n; i++) {
            tempList.add(currentChar);
            String result = dfsFindDifferentBinaryString(hashSet, tempList, n, currentChar);
            if (result != "") {
                return result;
            }

            tempList.remove(tempList.size() - 1);
            if (currentChar == '0') {
                currentChar = '1';
            } else {
                currentChar = '0';
            }
        }

        return "";
    }

    public static List<List<Integer>> combinationSum3(int k, int n) {
        List<List<Integer>> result = new ArrayList<>();
        dfsCombinationSum3(result, new ArrayList<>(), 1, 0, k, n);
        return result;
    }

    static void dfsCombinationSum3(List<List<Integer>> result,
                                   List<Integer> tempList,
                                   int index,
                                   int currentSum,
                                   int k,
                                   int n) {
        if (tempList.size() == k && currentSum == n) {
            result.add(new ArrayList<>(tempList));
        }

        for (int i = index; i <= 9; i++) {
            tempList.add(i);
            currentSum += i;
            dfsCombinationSum3(result, tempList, i + 1, currentSum, k, n);
            currentSum -= i;
            tempList.remove(tempList.size() - 1);
        }
    }

    public static List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> result = new ArrayList<>();
        dfsCombine(result, new ArrayList(), n, k, 1);
        return result;
    }

    static void dfsCombine(List<List<Integer>> result, List<Integer> tempList, int n, int k, int index) {
        if (tempList.size() == k) {
            result.add(new ArrayList<>(tempList));
            return;
        }

        for (int i = index; i <= n; i++) {

            tempList.add(i);
            dfsCombine(result, tempList, n, k, i + 1);
            tempList.remove(tempList.size() - 1);
        }
    }

    public static List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> result = new ArrayList<>();

        dfsCombinationSum(result, candidates, new ArrayList(), 0, target, 0);
        return result;
    }

    static void dfsCombinationSum(List<List<Integer>> result,
                                  int[] candidates,
                                  List<Integer> tempList,
                                  int currentSum,
                                  int target,
                                  int index) {
        if (currentSum > target) {
            return;
        }
        if (currentSum == target) {
            result.add(new ArrayList<>(tempList));
            return;
        }

        for (int i = index; i < candidates.length; i++) {
            tempList.add(candidates[i]);
            currentSum += candidates[i];
            dfsCombinationSum(result, candidates, tempList, currentSum, target, i);
            tempList.remove(tempList.size() - 1);
            currentSum -= candidates[i];
        }
    }

    static int getMaximumGoldMax = -1;

    static int getMaximumGold(int[][] grid) {
        boolean[][] visited = new boolean[grid.length][];
        for (int i = 0; i < visited.length; i++) {
            visited[i] = new boolean[grid[0].length];
        }

        dfsGrid(grid, visited, 0, 0, 0);
        return getMaximumGoldMax;
    }

    static void dfsGrid(int[][] grid, boolean[][] visited, int x, int y, int currentSum) {

        for (int i = x; i < grid.length; i++) {
            for (int j = y; j < grid[i].length; j++) {
                if (visited[i][j] || grid[i][j] == 0) {
                    continue;
                }

                currentSum += grid[i][j];
                getMaximumGoldMax = Math.max(getMaximumGoldMax, currentSum);

                visited[i][j] = true;
                dfsGrid(grid, visited, x + 1, y, currentSum);
                dfsGrid(grid, visited, x - 1, y, currentSum);
                dfsGrid(grid, visited, x, y + 1, currentSum);
                dfsGrid(grid, visited, x, y - 1, currentSum);
                visited[i][j] = false;

            }
        }
    }

    public static List<String> letterCasePermutation(String s) {
        List<String> res = new ArrayList<>();
        char[] a = s.toCharArray();
        letterDfs(a, 0, res);
        return res;
    }

    static void letterDfs(char[] letters, int pos, List<String> result) {
        if (pos == letters.length) {
            result.add(new String(letters));
            return;
        }

        if (!Character.isDigit(letters[pos])) {
            letters[pos] = Character.toUpperCase(letters[pos]);
            letterDfs(letters, pos + 1, result);
            letters[pos] = Character.toLowerCase(letters[pos]);
            letterDfs(letters, pos + 1, result);
        } else {
            letterDfs(letters, pos + 1, result);
        }
    }

    public static List<List<Integer>> subsetsWithDup(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        boolean[] visited = new boolean[nums.length];
        Arrays.sort(nums);
        generateSubsetsWithDuplicate(result, nums, new ArrayList<>(), visited, 0);

        return result;
    }

    public static void generateSubsetsWithDuplicate(List<List<Integer>> result,
                                                    int[] nums,
                                                    List<Integer> tempList,
                                                    boolean[] visited,
                                                    int start) {
        result.add(new ArrayList<>(tempList));
        for (int i = start; i < nums.length; i++) {
            if (i > start && nums[i - 1] == nums[i] || visited[i]) {
                continue;
            }

            visited[i] = true;
            tempList.add(nums[i]);
            generateSubsetsWithDuplicate(result, nums, tempList, visited, i + 1);
            visited[i] = false;
            tempList.remove(tempList.size() - 1);
        }
    }

    public static List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        boolean[] visited = new boolean[nums.length];
        generateSubsets(result, 0, nums, visited, new ArrayList<>());
        return result;
    }

    public static void generateSubsets(List<List<Integer>> result,
                                       int start,
                                       int[] nums,
                                       boolean[] visited,
                                       List<Integer> tempList) {
        result.add(new ArrayList<>(tempList));

        for (int i = start; i < nums.length; i++) {
            if (visited[i]) {
                continue;
            }
            tempList.add(nums[i]);
            visited[i] = true;
            generateSubsets(result, i + 1, nums, visited, tempList);
            visited[i] = false;
            tempList.remove(tempList.size() - 1);
        }
    }

    public static int numTilePossibilities(String tiles) {
        HashSet<String> hashSet = new HashSet<>();
        boolean[] visited = new boolean[tiles.length()];

        recursiveSubstrings(0, visited, tiles, hashSet, "");
        return (int) hashSet.stream().count();
    }

    public static void recursiveSubstrings(int start,
                                           boolean[] visited,
                                           String tiles,
                                           HashSet<String> hashSet,
                                           String tempString) {
        hashSet.add(tempString);
        for (int i = 0; i < tiles.length(); i++) {
            if (visited[i]) {
                continue;
            }
            tempString += tiles.charAt(i);
            visited[i] = true;
            recursiveSubstrings(i + 1, visited, tiles, hashSet, tempString);
            visited[i] = false;
            tempString = tempString.substring(0, tempString.length() - 1);
        }
    }

    public static List<List<Integer>> permutations(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        boolean[] used = new boolean[nums.length];
        dfs(result, new ArrayList<>(), nums, used);
        return result;
    }

    public static void dfs(List<List<Integer>> result, List<Integer> holdTempResult, int[] nums, boolean[] used) {
        if (holdTempResult.size() == nums.length) {
            result.add(new ArrayList<>(holdTempResult));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (used[i]) {
                continue;
            }
            used[i] = true;
            holdTempResult.add(nums[i]);
            dfs(result, holdTempResult, nums, used);
            used[i] = false;
            holdTempResult.remove(holdTempResult.size() - 1);
        }
    }

    public static int minFlipsMonoIncr(String s) {
        // 010110
        int zero = 0, ones = 0, ans = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '0') {
                zero++;
            } else {
                ones++;
            }

            if (zero > ones) {
                ans += ones;
                zero = 0;
                ones = 0;
            }
        }

        return ans + zero;
    }

    public static int numSubmat(int[][] mat) {

        int x = mat.length, y = mat[0].length, result = 0;

        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                result += calculateSubmatrx(mat, i, j);
            }
        }

        return result;
    }

    public static int calculateSubmatrx(int[][] matrix, int a, int b) {
        int count = 0;
        int rightBound = matrix[0].length;

        for (int i = a; i < matrix.length; i++) {
            for (int j = b; j < rightBound; j++) {
                if (matrix[i][j] == 0) {
                    rightBound = j;
                } else {
                    count++;
                }
            }
        }

        return count;
    }

    public static int findTheWinner(int n, int k) {
        LinkedList<Integer> linkedList = new LinkedList<>();
        // add to linked list
        for (int i = 1; i <= n; i++) {
            linkedList.add(i);
        }

        // traverse the linked list
        int i = 0;
        while (linkedList.size() != 1) {
            int counter = k;
            while (counter != 0) {
                counter--;
                if (counter == 0) {
                    linkedList.remove(i);
                    if (i >= linkedList.size()) {
                        i = 0;
                    }
                    break;
                }

                i++;
                if (i >= linkedList.size()) {
                    i = 0;
                }
            }
            // remove that element
        }

        return linkedList.get(0);
    }

    public static int maxProfitWithFee(int[] prices, int fee) {
        var n = prices.length;
        if (n <= 1) {
            return 0;
        }
        int[] buy = new int[n], sell = new int[n];
        buy[0] = -prices[0] - fee;
        sell[0] = 0;
        for (int i = 1; i < n; i++) {
            buy[i] = Math.max(buy[i - 1], sell[i - 1] - prices[i] - fee);
            sell[i] = Math.max(sell[i - 1], buy[i - 1] + prices[i]);
        }
        return sell[n - 1];
    }

    //https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/
    public static int maxProfit2(int[] prices) { //
        int result = 0, n = prices.length - 1;
        int i = 0;
        while (i < n) {
            int sell = 0, buy = 0;
            //best time to buy
            while (i < n && prices[i] >= prices[i + 1]) {
                i++;
            }
            buy = prices[i];

            while (i < n && prices[i] < prices[i + 1]) {
                i++;
            }
            sell = prices[i];

            result += sell - buy;
            // best time to sell
        }
        return result;
    }

    public static int maxProfit(int[] prices) {
        int max = 0, currentMin = prices[0];
        for (int i = 0; i < prices.length; i++) {
            if (prices[i] > currentMin) {
                max = Math.max(max, prices[i] - currentMin);
            } else {
                currentMin = prices[i];
            }
        }
        return max;
    }

    public static int maxProfit(int[] prices, int fee) {
        return 1;
    }

    // 2,1,6,4
    public static int waysToMakeFair(int[] nums) {
        int numsSum = Arrays.stream(nums).sum();
        int count = 0;
        boolean odd = false;
        for (int i = 0; i < nums.length; i++) {
            int sum = 0;
            for (int j = 0; j < nums.length; j++) {
                if (j == i) {
                    continue;
                }
                if (odd) {
                    sum += nums[j];
                }
                odd = !odd;
            }
            if (sum == (numsSum - nums[i]) - sum) {
                count++;
            }
        }

        return count;
    }

    public static boolean checkIfExist(int[] arr) {
        Set<Integer> set = new HashSet<>();

        for (int i = 0; i < arr.length; i++) {
            int num = arr[i];  // 14
            if (((double) num / 2 == (int) num / 2 && set.contains(num / 2)) || set.contains(num * 2)) {
                return true;
            }
            set.add(num);
        }

        return false;
    }

    public static boolean isLongPressedName(String name, String typed) {
        int typedPosition = 0;
        int namePosition = 0;
        if (name.length() > typed.length()) {
            return false;
        }

        while (typedPosition < typed.length() || namePosition < name.length()) {

            // s
            Character c = typed.charAt(typedPosition);
            int typedCounter = 0;
            while (typed.length() > typedPosition && typed.charAt(typedPosition) == c) {
                typedCounter++;
                typedPosition++;
            }

            int nameCounter = 0;
            while (name.length() > namePosition && name.charAt(namePosition) == c) {
                nameCounter++;
                namePosition++;
            }

            if (nameCounter > typedCounter || nameCounter == 0 || (namePosition < name.length() && typedPosition == typed.length())) {
                return false;
            }
        }

        return true;
    }

    public static int N;

    public static List<String> generateParenthesis(int n) {
        N = n;
        var result = new ArrayList<String>();
        rec("(", result, 1, 0);
        return result;
    }

    // (    ((  ()
    public static void rec(String sub, List<String> parenthesis, int open, int closed) {
        if (sub.length() == N * 2) {
            parenthesis.add(sub);
            return;
        }
        // (()
        if (open > closed) {
            if (open < N) {
                rec(sub + '(', parenthesis, open + 1, closed);
            }
            if (closed < N) {
                rec(sub + ')', parenthesis, open, closed + 1);
            }
        }
        // ()()(
        if (open == closed) {

            rec(sub + '(', parenthesis, open + 1, closed);
        }
    }

    public static int getKth(int lo, int hi, int k) {
        //   PriorityQueue<Item> pq = new PriorityQueue<>((a, b) -> (a.y == b.y) ? (b.x - a.x) : (b.y - a.y));
        // TreeSet<Item> set = new TreeSet<>((a,b)-> b.y!=a.y ? a.y-b.y:a.x-b.x);
        TreeSet<Item> set = new TreeSet<>(new Comparator<Item>() {
            @Override
            public int compare(Item o1, Item o2) {
                if (o1.y == o2.y) {
                    return o1.x.compareTo(o2.x);
                } else {
                    return o1.y.compareTo(o2.y);
                }
            }
        });
        while (lo <= hi) {
            var item = lo;
            var counter = 0;
            while (item != 1) {
                if (item % 2 == 0) {
                    item /= 2;
                } else {
                    item = (item * 3) + 1;
                }
                counter++;
            }
            set.add(new Item(lo, counter));
            lo++;
        }
        int res = 0;
        int counter = 0;
        for (Item item : set) {
            //  System.out.println(movie);
            res = item.x;
            counter++;
            if (counter == k) {
                break;
            }
        }
        return res;
    }

    public static int countVowelStrings(int n) {
        int a = 1, e = 1, i = 1, o = 1, u = 1;
        for (int c = 1; c < n; c++) {
            a = a + e + i + o + u;
            e = e + i + o + u;
            i = i + o + u;
            o = o + u;
            u = u;
        }
        return a + e + i + o + u;
    }

    public static int[] executeInstructions(int n, int[] startPos, String s) {
        // n = 3, startPos = [0,1], s = "RRDDLU"
        int len = s.length();
        int[] result = new int[len];
        int charPosition = 0;
        while (charPosition < len) {
            int counter = 0;

            int i = startPos[0];
            int j = startPos[1];
            int innerCharPosition = charPosition;
            while (innerCharPosition < len) {
                if (s.charAt(innerCharPosition) == 'R') {
                    j++;
                } else if (s.charAt(innerCharPosition) == 'D') {
                    i++;
                } else if (s.charAt(innerCharPosition) == 'L') {
                    j--;
                } else if (s.charAt(innerCharPosition) == 'U') {
                    i--;
                }
                if (i >= n || j >= n || i < 0 || j < 0) {
                    break;
                }
                counter++;
                innerCharPosition++;
            }

            result[charPosition] = counter;
            charPosition++;
        }
        return result;
    }

    public static String destCity(List<List<String>> paths) {
        HashSet<String> set = new HashSet<>();
        for (int i = 0; i < paths.size(); i++) {
            set.add(paths.get(i).get(0));
        }
        for (int i = 0; i < paths.size(); i++) {
            if (!set.contains(paths.get(i).get(1))) {
                return paths.get(i).get(1);
            }
        }
        return "";
    }

    public static int countSquares(int[][] matrix) {
        int res = 0;

        for (int i = 0; i < matrix[0].length; i++) {
            if (matrix[0][i] == 1) {
                res++;
            }
        }
        for (int i = 1; i < matrix.length; i++) {
            if (matrix[i][0] == 1) {
                res++;
            }
        }

        for (int i = 1; i < matrix.length; i++) {
            for (int j = 1; j < matrix[i].length; j++) {
                if (matrix[i][j] == 0) {
                    continue;
                }
                int top = matrix[i - 1][j], left = matrix[i][j - 1], diagonal = matrix[i - 1][j - 1];
                int val = Math.min(top, Math.min(left, diagonal)) + 1;
                matrix[i][j] = val;
                res += val;
            }
        }

        return res;
    }

    public static int numTeams(int[] rating) {

        //[2,5,3,4,1]
        int res = 0;
        for (int i = 1; i < rating.length - 1; i++) {
            // go left
            int left = i - 1, right = i + 1;
            int leftSmallerCounter = 0, leftLargerCounter = 0, rightSmallerCounter = 0, rightLargerCounter = 0;
            while (left >= 0) {
                if (rating[i] > rating[left]) {
                    leftSmallerCounter++;
                }
                if (rating[i] < rating[left]) {
                    leftLargerCounter++;
                }
                left--;
            }
            while (right < rating.length) {
                if (rating[i] < rating[right]) {
                    rightLargerCounter++;
                }
                if (rating[i] > rating[right]) {
                    rightSmallerCounter++;
                }
                right++;
            }
            if (leftSmallerCounter != 0 && rightLargerCounter != 0) {
                res += leftSmallerCounter * rightLargerCounter;
            }
            if (leftLargerCounter != 0 && rightSmallerCounter != 0) {
                res += leftLargerCounter * rightSmallerCounter;
            }
        }
        return res;
    }

    public static int numberOfArithmeticSlices(int[] nums) {
        // 12345

        // 3-2 == 2-1
        int sum = 0;
        int cur = 0;
        for (int i = 2; i < nums.length; i++) {
            if (nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]) {
                cur += 1;
                sum += cur;
            } else {
                cur = 0;
            }
        }

        return sum;
    }

    public static String longestPalindrome(String s) {
        int l = s.length();
        if (l == 1) {
            return s;
        }
        if (l == 2) {
            return s.substring(0, 1);
        }

        int max = 0;
        String res = "" + s.charAt(0);
        boolean[][] dp = new boolean[l][l];
        for (int i = l - 1; i >= 0; i--) {
            dp[i][i] = true;
            for (int j = i + 1; j < l; j++) {
                if (s.charAt(i) == s.charAt(j) && (dp[i + 1][j - 1] || j - i == 1)) {
                    dp[i][j] = true;
                    int dif = j - i + 1;
                    if (max < dif) {
                        max = dif;
                        res = s.substring(i, j + 1);
                    }
                }
            }
        }
        return res;
    }

    public static int longestPalindromeSubseq(String s) {
        int l = s.length();
        int[][] dp = new int[l][l];
        for (int i = l - 1; i >= 0; i--) {
            dp[i][i] = 1;
            for (int j = i + 1; j < l; j++) {
                if (s.charAt(i) == s.charAt(j)) {
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                } else {
                    dp[i][j] = Math.max(dp[i][j - 1], dp[i + 1][j]);
                }
            }

        }
        return dp[0][l - 1];
    }

    private static int isPalindrome(char[] arr, int i, int j) {
        int count = 0;
        while (i >= 0 && j < arr.length && arr[i--] == arr[j++]) {
            count++;
        }
        return count;
    }

    public static int countSubstrings(String s) {
        if (s.length() == 0) {
            return 0;
        }
        int result = 0;
        char[] s_chr = s.toCharArray();
        for (int i = 0; i < s_chr.length; i++) {
            result += isPalindrome(s_chr, i, i);
            result += isPalindrome(s_chr, i, i + 1);
        }
        return result;
    }

    public static List<String> findAndReplacePattern(String[] words, String pattern) {
        // a - 0
        // b - 1,2
        Map<Character, ArrayList<Integer>> map = new HashMap<>();
        for (int i = 0; i < pattern.length(); i++) {
            if (map.containsKey(pattern.charAt(i))) {
                var arr = map.get(pattern.charAt(i));
                arr.add(i);
                map.put(pattern.charAt(i), arr);
            } else {
                var arr = new ArrayList<Integer>();
                arr.add(i);
                map.put(pattern.charAt(i), arr);
            }
        }

        for (int i = 0; i < words.length; i++) {
            // abc

        }

        Map<Character, ArrayList<Integer>> arr = new HashMap<>();

        return new ArrayList<>();
    }

    public static String frequencySort(String s) {

        HashMap<Character, Integer> map = new HashMap<>();

        for (int i = 0; i < s.length(); i++) {
            map.put(s.charAt(i), map.getOrDefault(s.charAt(i), 0) + 1);
        }
        PriorityQueue<Map.Entry<Character, Integer>> queue = new PriorityQueue<>(new Comparator<Map.Entry<Character, Integer>>() {
            @Override
            public int compare(Map.Entry<Character, Integer> o1, Map.Entry<Character, Integer> o2) {
                if (o1.getValue() > o2.getValue()) {
                    return -1;
                } else {
                    return 1;
                }
            }
        });

        for (var n : map.entrySet()) {
            queue.add(n);
        }

        String result = "";
        while (queue.size() > 0) {
            var item = queue.poll();
            var key = item.getKey();
            for (int i = 0; i < item.getValue(); i++) {
                result += item.getKey();
            }
        }

        return result;
    }

    public static String reversePrefix(String word, char ch) {
        String result = "";
        boolean trigger = false;
        int t = -1;
        for (int i = 0; i < word.length(); i++) {
            result = word.charAt(i) + result;
            if (word.charAt(i) == ch) {
                t = i;
                break;
            }
        }
        if (t == -1) {
            return word;
        }

        var tt = result + word.substring(t + 1);
        return tt;
    }

    public static boolean halvesAreAlike(String s) {
        Set<Character> ch = new HashSet<>(Arrays.asList('a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'));
        int left = 0, right = s.length() - 1;
        int leftCount = 0;
        int rightCount = 0;
        while (left < right) {
            if (ch.contains(s.charAt(left))) {
                leftCount++;
            }
            if (ch.contains(s.charAt(right))) {
                rightCount++;
            }
            left++;
            right--;
        }
        return leftCount == rightCount;
    }

    public static int numberOfBeams(String[] bank) {
        int result = 0;
        int lastCount = 0;
        for (int i = 0; i < bank.length; i++) {
            int deviceCount = (int) bank[i].chars().filter(ch -> ch == '1').count();

            if (i == 0) {
                lastCount = deviceCount;
                continue;
            }
            if (deviceCount == 0) {
                continue;
            }

            result += deviceCount * lastCount;
            lastCount = deviceCount;
        }
        return result;
    }

    public static int countCharacters(String[] words, String chars) {
        // ["cat","bt","hat","tree"], chars = "atach"
        HashMap<Character, Integer> map = new HashMap<>();
        int result = 0;
        for (int i = 0; i < chars.length(); i++) {
            map.put(chars.charAt(i), map.getOrDefault(chars.charAt(i), 0) + 1);
        }

        for (int i = 0; i < words.length; i++) {

            HashMap<Character, Integer> tempMap = new HashMap<>(map);
            boolean trigger = false;
            for (int j = 0; j < words[i].length(); j++) {
                var c = words[i].charAt(j);
                if (!tempMap.containsKey(c)) {
                    trigger = true;
                    break;
                } else {
                    var valCount = tempMap.get(c);
                    if (valCount == 1) {
                        tempMap.remove(c);
                    } else {
                        tempMap.put(c, tempMap.getOrDefault(c, 0) - 1);
                    }
                }
            }
            if (!trigger) {
                result += words[i].length();
            }
        }
        return result;
    }

    public static List<String> commonChars(String[] words) {
        HashMap<Character, Integer> map = new HashMap<>();
        ArrayList<String> res = new ArrayList<>();

        for (int i = 0; i < words[0].length(); i++) {
            var c = words[0].charAt(i);
            map.put(c, map.getOrDefault(words[0].charAt(i), 0) + 1);
        }

        for (int i = 1; i < words.length; i++) {
            // have max
            HashMap<Character, Integer> next = new HashMap<>();

            for (int j = 0; j < words[i].length(); j++) {
                var c = words[i].charAt(j);
                if (map.containsKey(c) && map.get(c) > 0) {
                    map.put(c, map.getOrDefault(c, 0) - 1);
                    next.put(c, next.getOrDefault(c, 0) + 1);
                }
            }
            map = next;
        }

        for (var item : map.entrySet()) {
            var val = item.getValue();
            for (int i = 0; i < val; i++) {
                res.add(item.getKey().toString());
            }
        }

        return res;
    }

    public static int[] relativeSortArray(int[] arr1, int[] arr2) {

        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < arr1.length; i++) {
            map.put(arr1[i], map.getOrDefault(arr1[i], 0) + 1);
        }

        ArrayList<Integer> arrayList = new ArrayList<>();
        for (int i = 0; i < arr2.length; i++) {
            var key = arr2[i];
            if (map.containsKey(key)) {
                for (int j = 0; j < map.get(key); j++) {
                    arrayList.add(key);
                }
                map.remove(key);
            }
        }

        PriorityQueue<Integer> q = new PriorityQueue<>(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                if (o1 > o2) {
                    return 1;
                } else {
                    return -1;
                }
            }
        });
        for (var item : map.entrySet()) {
            int c = item.getValue();

            for (int j = 0; j < c; j++) {
                q.add(item.getKey());
            }
        }

        while (q.size() != 0) {
            var item = q.poll();
            arrayList.add(item);
        }

        int[] result = new int[arrayList.size()];
        for (int i = 0; i < arrayList.size(); i++) {
            result[i] = arrayList.get(i);
        }

        return result;
    }

    public static String[] uncommonFromSentences(String s1, String s2) {
        HashMap<String, Integer> map1 = new HashMap<>();
        HashMap<String, Integer> map2 = new HashMap<>();

        String[] arr1 = s1.split(" ");
        String[] arr2 = s2.split(" ");
        ArrayList<String> list = new ArrayList<>();

        for (int i = 0; i < arr1.length; i++) {
            map1.put(arr1[i], map1.getOrDefault(arr1[i], 0) + 1);
        }
        for (int i = 0; i < arr2.length; i++) {
            map2.put(arr2[i], map2.getOrDefault(arr2[i], 0) + 1);
        }
        for (int i = 0; i < arr1.length; i++) {
            var item = arr1[i];
            if (!map2.containsKey(item) && map1.get(item) == 1) {
                list.add(item);
            }
        }
        for (int i = 0; i < arr2.length; i++) {
            var item = arr2[i];
            if (!map1.containsKey(item) && map2.get(item) == 1) {
                list.add(item);
            }
        }
        String[] result = new String[list.size()];
        for (int i = 0; i < list.size(); i++) {
            result[i] = list.get(i);
        }

        return result;
    }

    public static boolean isPathCrossing(String path) {
        HashSet<Item> set = new HashSet<>();
        set.add(new Item(0, 0)); // add initial position

        int x = 0, y = 0;

        for (int i = 0; i < path.length(); i++) {
            if (path.charAt(i) == 'N') {
                y++;
            } else if (path.charAt(i) == 'S') {
                y--;
            } else if (path.charAt(i) == 'W') {
                x--;
            } else {
                x++;
            }

            if (set.contains(new Item(x, y))) {
                return true;
            } else {
                set.add(new Item(x, y));
            }
        }

        return false;
    }

    static class Item {

        private Integer x;
        private Integer y;

        public Item(Integer x, Integer y) {
            this.x = x;
            this.y = y;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) {
                return true;
            }
            if (o == null || getClass() != o.getClass()) {
                return false;
            }
            Item item = (Item) o;
            return Objects.equals(x, item.x) && Objects.equals(y, item.y);
        }

        @Override
        public int hashCode() {
            return Objects.hash(x, y);
        }
    }

    public static boolean isAlienSorted(String[] words, String order) {
        if (words.length == 1) {
            return true;
        }
        HashMap<Character, Integer> map = new HashMap<>();
        int counter = 26;
        for (int i = 0; i < order.length(); i++) {
            map.put(order.charAt(i), counter--);
        }

        for (int i = 1; i < words.length; i++) {
            boolean trigger = false;
            int position = 0;
            while (words[i].length() > position || words[i - 1].length() > position) {
                if (map.get(words[i].charAt(position)) > map.get(words[i - 1].charAt(position))) {
                    return false;
                } else if (map.get(words[i].charAt(position)) < map.get(words[i - 1].charAt(position))) {
                    trigger = true;
                    break;
                }
                position++;
            }
            if (words[i - 1].length() > position && !trigger) {
                return false;
            }
        }

        return true;
    }

    public static boolean isCovered(int[][] ranges, int left, int right) {
        for (int i = left; i <= right; i++) {
            boolean trigger = false;
            for (int j = 0; j < ranges.length; j++) {
                if (i >= ranges[j][0] || i <= ranges[j][1]) {
                    trigger = true;
                    break;
                }
            }
            if (!trigger) {
                return false;
            }
        }

        return true;
    }

    public static boolean areAlmostEqual(String s1, String s2) {
        Set<Character> set1 = new HashSet<>();
        Set<Character> set2 = new HashSet<>();

        Integer counter = 0;
        for (int i = 0; i < s1.length(); i++) {
            if (s1.charAt(i) != s2.charAt(i)) {

                if (++counter > 2) {
                    return false;
                }

                set1.add(s1.charAt(i));
                set2.add(s2.charAt(i));
            }
        }
        for (Character c : set1) {
            if (!set2.contains(c)) {
                return false;
            }
        }

        return true;
    }

    public int[] findErrorNums(int[] nums) {
        HashSet<Integer> set = new HashSet<>();
        Integer twice = -1;
        Integer missing = -1;

        for (int i = 0; i < nums.length; i++) {
            if (i + 1 != nums[i]) {
                missing = i + 1;
            }
            if (set.contains(nums[i])) {
                twice = nums[i];
            }
            set.add(nums[i]);
        }

        for (int i = 1; i < nums.length + 1; i++) {
            if (!set.contains(i)) {
                missing = i;
            }
        }

        int[] res = new int[2];
        res[0] = twice;
        res[1] = missing;
        return res;
    }

    public static String mostCommonWord(String paragraph, String[] banned) {
        HashMap<String, Integer> hash = new HashMap<>();
        Set<String> set = new HashSet<>();

        for (int i = 0; i < banned.length; i++) {
            set.add(banned[i]);
        }

        String[] arr = paragraph.replaceAll("\\W+", " ").toLowerCase().split("\\s+");

        for (int i = 0; i < arr.length; i++) {
            String curr = arr[i].toLowerCase();
            if (set.contains(curr)) {
                continue;
            }

            if (!hash.containsKey(curr)) {
                hash.put(curr, 1);
            } else {
                hash.put(curr, hash.get(curr) + 1);
            }
        }

        PriorityQueue<Map.Entry<String, Integer>> priorityQueue = new PriorityQueue<>(new Comparator<Map.Entry<String, Integer>>() {
            @Override
            public int compare(Map.Entry<String, Integer> o1, Map.Entry<String, Integer> o2) {

                if (o1.getValue() > o2.getValue()) {
                    return -1;
                } else {
                    return 1;
                }
            }
        });

        for (Map.Entry<String, Integer> entry : hash.entrySet()) {
            priorityQueue.add(entry);
        }

        return priorityQueue.peek().getKey();
    }

    public int[] intersect(int[] nums1, int[] nums2) {
        // nums1 = [4,9,5], nums2 = [9,4,9,8,4]  -> Output: [4,9]

        HashMap<Integer, Integer> hash = new HashMap<>();
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < nums1.length; i++) {
            if (!hash.containsKey(nums1[i])) {
                hash.put(nums1[i], 1);
            } else {
                hash.put(nums1[i], hash.get(nums1[i]) + 1);
            }
        }

        for (int i = 0; i < nums2.length; i++) {
            if (hash.containsKey(nums2[i]) && hash.get(nums2[i]) > 0) {
                list.add(nums2[i]);
                hash.put(nums2[i], hash.get(nums2[i]) - 1);
            }
        }
        int[] arr = new int[list.size()];

        for (int i = 0; i < arr.length; i++) {
            arr[i] = list.get(i);
        }

        return arr;
    }

    public static boolean isIsomorphic(String s, String t) {
        if (s.length() != t.length()) {
            return false;
        }

        Map<Character, Character> map = new HashMap<>();
        Set<Character> set = new HashSet<>();

        for (int i = 0; i < s.length(); i++) {
            if (!map.containsKey(s.charAt(i))) {
                if (set.contains(t.charAt(i))) {
                    return false;
                }
                map.put(s.charAt(i), t.charAt(i));
                set.add(t.charAt(i));
            } else {
                if (map.get(s.charAt(i)) != t.charAt(i)) {
                    return false;
                }
            }
        }

        return true;
    }

}

