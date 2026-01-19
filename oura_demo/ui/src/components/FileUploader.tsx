import React, { useCallback, useRef, useState } from 'react';
import type { ColumnValidationResult } from '../types';

interface FileUploaderProps {
  onUpload: (file: File) => void;
  isLoading: boolean;
  disabled: boolean;
  validation: ColumnValidationResult | null;
  uploadedFileName: string | null;
}

export const FileUploader: React.FC<FileUploaderProps> = ({
  onUpload,
  isLoading,
  disabled,
  validation,
  uploadedFileName,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);

      if (disabled || isLoading) return;

      const files = e.dataTransfer.files;
      if (files.length > 0) {
        onUpload(files[0]);
      }
    },
    [onUpload, disabled, isLoading]
  );

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled && !isLoading) {
      setIsDragging(true);
    }
  }, [disabled, isLoading]);

  const handleDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (files && files.length > 0) {
        onUpload(files[0]);
      }
    },
    [onUpload]
  );

  const handleClick = () => {
    if (!disabled && !isLoading) {
      fileInputRef.current?.click();
    }
  };

  return (
    <div className="card p-4 animate-fade-in">
      <div className="flex items-center gap-2 mb-3">
        <div 
          className="w-6 h-6 rounded-md flex items-center justify-center text-xs font-bold"
          style={{ background: 'rgba(126, 184, 218, 0.2)', color: '#7eb8da' }}
        >
          2
        </div>
        <h2 className="text-sm font-medium" style={{ color: '#f5f2ed' }}>Upload Data</h2>
      </div>

      <div
        onClick={handleClick}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={`
          relative border-2 border-dashed rounded-lg p-4 text-center cursor-pointer
          transition-all duration-200 ease-out
          ${isLoading ? 'opacity-50' : ''}
        `}
        style={{
          background: disabled 
            ? 'rgba(30, 27, 24, 0.4)' 
            : isDragging
              ? 'rgba(126, 184, 218, 0.1)'
              : 'rgba(30, 27, 24, 0.5)',
          borderColor: disabled
            ? 'rgba(140, 125, 105, 0.2)'
            : isDragging
              ? 'rgba(126, 184, 218, 0.5)'
              : 'rgba(140, 125, 105, 0.3)',
          cursor: disabled ? 'not-allowed' : 'pointer',
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".parquet,.csv"
          onChange={handleFileSelect}
          disabled={disabled || isLoading}
          className="hidden"
        />

        {isLoading ? (
          <div className="flex items-center justify-center gap-2">
            <svg className="w-5 h-5 animate-spin" style={{ color: '#7eb8da' }} fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            <span className="text-xs" style={{ color: 'rgba(245, 242, 237, 0.6)' }}>Processing...</span>
          </div>
        ) : uploadedFileName ? (
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg flex-shrink-0 flex items-center justify-center" style={{ background: 'rgba(7, 173, 152, 0.2)' }}>
              <svg className="w-4 h-4" style={{ color: '#07ad98' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <div className="text-left min-w-0">
              <p className="text-xs font-medium truncate" style={{ color: '#f5f2ed' }}>{uploadedFileName}</p>
              <p className="text-xs" style={{ color: 'rgba(245, 242, 237, 0.4)' }}>Click to change</p>
            </div>
          </div>
        ) : (
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg flex-shrink-0 flex items-center justify-center" style={{ background: disabled ? 'rgba(60, 55, 50, 0.4)' : 'rgba(140, 125, 105, 0.3)' }}>
              <svg className="w-4 h-4" style={{ color: disabled ? 'rgba(245, 242, 237, 0.3)' : '#f5f2ed' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
            </div>
            <div className="text-left">
              <p className="text-xs" style={{ color: disabled ? 'rgba(245, 242, 237, 0.3)' : '#f5f2ed' }}>
                {disabled ? 'Select dataset first' : 'Drop file or click'}
              </p>
              {!disabled && (
                <p className="text-xs" style={{ color: 'rgba(245, 242, 237, 0.4)' }}>.parquet, .csv</p>
              )}
            </div>
          </div>
        )}
      </div>

      {validation && (
        <div 
          className="mt-3 p-2 rounded-lg flex items-center gap-2"
          style={{
            background: validation.valid ? 'rgba(7, 173, 152, 0.15)' : 'rgba(229, 115, 115, 0.15)',
          }}
        >
          {validation.valid ? (
            <>
              <svg className="w-4 h-4" style={{ color: '#07ad98' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
              <span className="text-xs" style={{ color: '#07ad98' }}>All columns valid</span>
            </>
          ) : (
            <>
              <svg className="w-4 h-4" style={{ color: '#e57373' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span className="text-xs" style={{ color: '#e57373' }}>
                Missing: {validation.missing_columns.join(', ')}
              </span>
            </>
          )}
        </div>
      )}
    </div>
  );
};
