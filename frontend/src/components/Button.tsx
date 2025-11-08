import React from 'react';

interface ButtonProps {
  children: React.ReactNode;
  onClick?: () => void;
  variant?: 'primary' | 'secondary' | 'outline';
  icon?: React.ReactNode;
  className?: string;
  disabled?: boolean;
}

export const Button: React.FC<ButtonProps> = ({
  children,
  onClick,
  variant = 'primary',
  icon,
  className = '',
  disabled = false
}) => {
  const baseClasses = 'px-6 py-3 rounded-full font-semibold transition-all duration-300 flex items-center justify-center gap-2 shadow-md hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed';

  const variantClasses = {
    primary: 'bg-gradient-to-r from-[#2E7D32] to-[#4CAF50] text-white hover:scale-105',
    secondary: 'bg-[#81C784] text-white hover:scale-105',
    outline: 'border-2 border-[#2E7D32] text-[#2E7D32] hover:bg-[#2E7D32] hover:text-white'
  };

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`${baseClasses} ${variantClasses[variant]} ${className}`}
    >
      {icon && <span>{icon}</span>}
      {children}
    </button>
  );
};
