// frontend/src/lib/hooks/useAuth.ts
import { create } from 'zustand';
import { apiClient } from '../api';
import { useMutation, useQuery } from '@tanstack/react-query';
import toast from 'react-hot-toast';

interface User {
    sub: string;
    username: string;
    email: string;
    full_name: string;
    role: string;
    permissions: string[];
}

interface AuthState {
    user: User | null;
    isAuthenticated: boolean;
    setUser: (user: User | null) => void;
    logout: () => void;
}

export const useAuthStore = create<AuthState>((set) => ({
    user: null,
    isAuthenticated: !!localStorage.getItem('access_token'),
    setUser: (user) => set({ user, isAuthenticated: !!user }),
    logout: () => {
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
        set({ user: null, isAuthenticated: false });
    },
}));

export function useLogin() {
    const setUser = useAuthStore((s) => s.setUser);

    return useMutation({
        mutationFn: async (credentials: { username: string; password: string }) => {
            const formData = new URLSearchParams();
            formData.append('username', credentials.username);
            formData.append('password', credentials.password);

            const { data } = await apiClient.post('/auth/token', formData, {
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            });
            return data;
        },
        onSuccess: async (data) => {
            localStorage.setItem('access_token', data.access_token);
            if (data.refresh_token) {
                localStorage.setItem('refresh_token', data.refresh_token);
            }

            const { data: user } = await apiClient.get('/auth/me');
            setUser(user);
            toast.success('Welcome back!');
        },
        onError: () => {
            toast.error('Invalid credentials');
        },
    });
}

export function useCurrentUser() {
    const { setUser, logout, isAuthenticated } = useAuthStore();

    return useQuery<User, Error>({
        queryKey: ['currentUser'],
        queryFn: async () => {
            const { data } = await apiClient.get('/auth/me');
            setUser(data);
            return data as User;
        },
        enabled: isAuthenticated,
        retry: false,
    });
}