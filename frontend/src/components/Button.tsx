import React from 'react';

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  icon?: React.ReactNode;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'primary', size = 'md', icon, children, ...props }, ref) => {
    const baseClasses =
      'inline-flex items-center justify-center rounded-full font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2';

    const variantClasses = {
      primary:
        'bg-gradient-to-r from-[#2E7D32] to-[#4CAF50] text-white hover:opacity-90 focus:ring-[#4CAF50]',
      secondary:
        'bg-[#E8F5E9] text-[#2E7D32] hover:bg-[#D5EAD6] focus:ring-[#4CAF50]',
      outline:
        'border border-[#BDBDBD] bg-transparent text-[#546E7A] hover:bg-gray-100 focus:ring-[#BDBDBD]',
      ghost: 'hover:bg-gray-100 text-[#546E7A] focus:ring-gray-200',
    };

    const sizeClasses = {
      sm: 'px-4 py-2 text-sm',
      md: 'px-6 py-3 text-base',
      lg: 'px-8 py-4 text-lg',
    };


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
